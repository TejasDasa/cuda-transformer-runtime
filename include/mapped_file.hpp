#ifndef MAPPED_FILE_HPP
#define MAPPED_FILE_HPP

#include <cstddef>
#include <string>

class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const void* data() const;
    std::size_t size() const;

private:
    int fd_ = -1;
    void* data_ = nullptr;
    std::size_t size_ = 0;
};


#endif