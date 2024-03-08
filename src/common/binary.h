#pragma once

#include "common/io_item.h"

#include <string>
#include <vector>

namespace marian {

const static int BINARY_FILE_VERSION = 1;

namespace io {
namespace binary {

void loadItems(const void* current, std::vector<io::Item>& items, bool mapped = false);
void loadItems(const std::string& fileName, std::vector<io::Item>& items);

io::Item getItem(const void* current, const std::string& vName);
io::Item getItem(const std::string& fileName, const std::string& vName);

void saveItems(const std::string& fileName, const std::vector<io::Item>& items);

void convertItems(const std::vector<io::Item>& items, Type toType = marian::Type::float16);

void convertItems(const std::vector<io::Item>& items, const std::string& toType) {
  io::binary::convertItems(items, marian::typeFromString(toType));
}

void printItems(const std::vector<io::Item>& items);

}  // namespace binary
}  // namespace io
}  // namespace marian
