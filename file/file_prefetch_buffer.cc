//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "file/file_prefetch_buffer.h"

#include <algorithm>
#include <mutex>
#include <iostream>
#include <chrono>

#include "file/random_access_file_reader.h"
#include "monitoring/histogram.h"
#include "monitoring/iostats_context_imp.h"
#include "port/port.h"
#include "test_util/sync_point.h"
#include "util/random.h"
#include "util/rate_limiter.h"
#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {

extern std::shared_ptr<Logger> _logger;

static void BGReadahead(const IOOptions& opts,
                        RandomAccessFileReader* reader,
                        uint64_t offset, size_t n,
                        AlignedBuffer* buf) {
  Slice result;

  reader->Read(opts, offset, n, &result, buf->BufferStart(), nullptr, true);
  buf->Size(result.size());
}

Status FilePrefetchBuffer::Prefetch(const IOOptions& opts,
                                    RandomAccessFileReader* reader,
                                    uint64_t offset, size_t n,
                                    bool for_compaction) {
  if (!enable_ || reader == nullptr) {
    return Status::OK();
  }

  TEST_SYNC_POINT("FilePrefetchBuffer::Prefetch:Start");
  size_t alignment = reader->file()->GetRequiredBufferAlignment();
  size_t offset_ = static_cast<size_t>(offset);
  uint64_t rounddown_offset = Rounddown(offset_, alignment);
  uint64_t roundup_end = Roundup(offset_ + n, alignment);
  uint64_t roundup_len = roundup_end - rounddown_offset;
  assert(roundup_len >= alignment);
  assert(roundup_len % alignment == 0);

  // Check if requested bytes are in the existing buffer_.
  // If all bytes exist -- return.
  // If only a few bytes exist -- reuse them & read only what is really needed.
  //     This is typically the case of incremental reading of data.
  // If no bytes exist in buffer -- full pread.

  Status s;
  uint64_t chunk_offset_in_buffer = 0;
  uint64_t chunk_len = 0;
  bool copy_data_to_new_buffer = false;
  if (buffer_->CurrentSize() > 0 && offset >= buffer_offset_ &&
      offset <= buffer_offset_ + buffer_->CurrentSize()) {
    if (offset + n <= buffer_offset_ + buffer_->CurrentSize()) {
      // All requested bytes are already in the buffer. So no need to Read
      // again.
      return s;
    } else {
      // Only a few requested bytes are in the buffer. memmove those chunk of
      // bytes to the beginning, and memcpy them back into the new buffer if a
      // new buffer is created.
      chunk_offset_in_buffer =
          Rounddown(static_cast<size_t>(offset - buffer_offset_), alignment);
      chunk_len = buffer_->CurrentSize() - chunk_offset_in_buffer;
      assert(chunk_offset_in_buffer % alignment == 0);
      assert(chunk_len % alignment == 0);
      assert(chunk_offset_in_buffer + chunk_len <=
             buffer_offset_ + buffer_->CurrentSize());
      if (chunk_len > 0) {
        copy_data_to_new_buffer = true;
      } else {
        // this reset is not necessary, but just to be safe.
        chunk_offset_in_buffer = 0;
      }
    }
  }

  // Create a new buffer only if current capacity is not sufficient, and memcopy
  // bytes from old buffer if needed (i.e., if chunk_len is greater than 0).
  if (buffer_->Capacity() < roundup_len) {
    buffer_->Alignment(alignment);
    buffer_->AllocateNewBuffer(static_cast<size_t>(roundup_len),
                              copy_data_to_new_buffer, chunk_offset_in_buffer,
                              static_cast<size_t>(chunk_len));
  } else if (chunk_len > 0) {
    // New buffer not needed. But memmove bytes from tail to the beginning since
    // chunk_len is greater than 0.
    buffer_->RefitTail(static_cast<size_t>(chunk_offset_in_buffer),
                      static_cast<size_t>(chunk_len));
  }

  Slice result;
  size_t read_len = static_cast<size_t>(roundup_len - chunk_len);

  if (for_compaction) {
    uint64_t buff_offset = chunk_len + read_len;
    uint64_t file_offset = rounddown_offset + buff_offset;
    assert(buffer_switch_ == nullptr);

    buffer_switch_ = new AlignedBuffer;
    buffer_switch_->Alignment(alignment);
    buffer_switch_->AllocateNewBuffer(readahead_size_);
    thread_ = new std::thread(BGReadahead, opts, reader,
                              file_offset, buffer_switch_->Capacity(),
                              buffer_switch_);
  }

  s = reader->Read(opts, rounddown_offset + chunk_len, read_len, &result,
                   buffer_->BufferStart() + chunk_len, nullptr, for_compaction);
  if (!s.ok()) {
    return s;
  }

#ifndef NDEBUG
  if (result.size() < read_len) {
    // Fake an IO error to force db_stress fault injection to ignore
    // truncated read errors
    IGNORE_STATUS_IF_ERROR(Status::IOError());
  }
#endif
  buffer_offset_ = rounddown_offset;
  buffer_->Size(static_cast<size_t>(chunk_len) + result.size());
  return s;
}

void FilePrefetchBuffer::SwapBuffersAndReadahead(const IOOptions& opts) {
  AlignedBuffer* tmp;

  tmp = buffer_switch_;
  buffer_switch_ = buffer_;
  buffer_ = tmp;

  thread_ = new std::thread(BGReadahead, opts, file_reader_,
                            buffer_offset_ + buffer_->CurrentSize(),
                            buffer_switch_->Capacity(), buffer_switch_);
}

bool FilePrefetchBuffer::TryReadFromCacheForCompaction(const IOOptions& opts,
                                          uint64_t offset, size_t n,
                                          Slice* result, Status* /*status*/) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // First time for this compaction
  if (!buffer_->CurrentSize()) {
    Prefetch(opts, file_reader_, offset, std::max(n, readahead_size_), true);
    *result = Slice(buffer_->BufferStart() + offset - buffer_offset_, n);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    ROCKS_LOG_INFO(_logger, "readahead: (1) prefetch %lu µs",
                   std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
  } else if (offset < buffer_offset_ + buffer_->CurrentSize() &&
             offset + n > buffer_offset_ + buffer_->CurrentSize()) {
    // Now, we should switch the buffer to the new one with triggering a new
    // thread to read the next data.  In this situation, we can guarantee that
    // the switch buffer is already ready.
    size_t alignment = file_reader_->file()->GetRequiredBufferAlignment();
    uint64_t chunk_offset_in_buffer =
          Rounddown(static_cast<size_t>(offset - buffer_offset_), alignment);
    uint64_t chunk_len = buffer_->CurrentSize() - chunk_offset_in_buffer;
    size_t roundup_len = Roundup(n, alignment);

    thread_->join();

    if (buffer_spanning_.Capacity() < roundup_len) {
      buffer_spanning_.Alignment(alignment);
      buffer_spanning_.AllocateNewBuffer(chunk_len + roundup_len);
    }

    memcpy(buffer_spanning_.BufferStart(),
           buffer_->BufferStart() + chunk_offset_in_buffer, chunk_len);
    memcpy(buffer_spanning_.BufferStart() + chunk_len,
           buffer_switch_->BufferStart(), roundup_len);

    *result = Slice(buffer_spanning_.BufferStart() +
                    (offset - buffer_offset_) - chunk_offset_in_buffer, n);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    ROCKS_LOG_INFO(_logger, "readahead: (2) spanning %lu µs",
                   std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    buffer_offset_ += buffer_->CurrentSize();
    if (buffer_switch_->CurrentSize() != buffer_switch_->Capacity()) {
      // In this case, the last chunk of the file.
      delete buffer_->Release();
      buffer_ = buffer_switch_;
      buffer_switch_ = nullptr;
    } else {
      SwapBuffersAndReadahead(opts);
    }

  } else if (offset == buffer_offset_ + buffer_->CurrentSize()) {
    thread_->join();
    buffer_offset_ += buffer_->CurrentSize();
    SwapBuffersAndReadahead(opts);
    *result = Slice(buffer_->BufferStart() + offset - buffer_offset_, n);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    ROCKS_LOG_INFO(_logger, "readahead: (3) swap %lu µs",
                   std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
  } else {
    // Now, it's time to trigger a new thread to read the next readahead block.
    *result = Slice(buffer_->BufferStart() + offset - buffer_offset_, n);
  }

  UpdateReadPattern(offset, n);
  return true;
}

bool FilePrefetchBuffer::TryReadFromCache(const IOOptions& opts,
                                          uint64_t offset, size_t n,
                                          Slice* result, Status* status,
                                          bool for_compaction) {
  if (track_min_offset_ && offset < min_offset_read_) {
    min_offset_read_ = static_cast<size_t>(offset);
  }
  if (!enable_ || offset < buffer_offset_) {
    return false;
  }

  if (for_compaction) {
    return TryReadFromCacheForCompaction(opts, offset, n, result, status);
  }

  // If the buffer contains only a few of the requested bytes:
  //    If readahead is enabled: prefetch the remaining bytes + readahead bytes
  //        and satisfy the request.
  //    If readahead is not enabled: return false.
  if (offset + n > buffer_offset_ + buffer_->CurrentSize()) {
    if (readahead_size_ > 0) {
      assert(file_reader_ != nullptr);
      assert(max_readahead_size_ >= readahead_size_);
      Status s;

      if (implicit_auto_readahead_) {
        // Prefetch only if this read is sequential otherwise reset
        // readahead_size_ to initial value.
        if (!IsBlockSequential(offset)) {
          UpdateReadPattern(offset, n);
          ResetValues();
          // Ignore status as Prefetch is not called.
          s.PermitUncheckedError();
          return false;
        }
        num_file_reads_++;
        if (num_file_reads_ <= kMinNumFileReadsToStartAutoReadahead) {
          UpdateReadPattern(offset, n);
          // Ignore status as Prefetch is not called.
          s.PermitUncheckedError();
          return false;
        }
      }
      s = Prefetch(opts, file_reader_, offset, n + readahead_size_,
                   for_compaction);
      if (!s.ok()) {
        if (status) {
          *status = s;
        }
#ifndef NDEBUG
        IGNORE_STATUS_IF_ERROR(s);
#endif
        return false;
      }
      readahead_size_ = std::min(max_readahead_size_, readahead_size_ * 2);
    } else {
      return false;
    }
  }

  UpdateReadPattern(offset, n);
  uint64_t offset_in_buffer = offset - buffer_offset_;
  *result = Slice(buffer_->BufferStart() + offset_in_buffer, n);
  return true;
}
}  // namespace ROCKSDB_NAMESPACE
