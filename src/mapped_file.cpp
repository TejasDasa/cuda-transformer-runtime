#include "mapped_file.hpp"
#include <string>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

const void* MappedFile::data() const
{
    return data_;
}

std::size_t MappedFile::size() const
{
    return size_;
}

MappedFile::MappedFile(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);

    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file.");
    }

    struct stat sb{};

    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Failed to get file status.");
    }

    if (sb.st_size <= 0) {
        close(fd_);
        throw std::runtime_error("Checkpoint file is empty.");
    }

    size_ = static_cast<std::size_t>(sb.st_size);

    void* mapped_addr = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);

    if (mapped_addr == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Memory mapped failed");
    }

    data_ = mapped_addr;
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        munmap(data_, size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}