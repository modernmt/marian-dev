#include <filesystem>
#include <regex>
#include "marian.h"
#include "common/binary.h"

using namespace marian;
namespace fs = std::filesystem;

bool is_fp16(fs::path from_model_meta) {
  std::string line;
  io::InputFileStream in(from_model_meta);
  bool model_section = false;

  while(getline(in, line)) {
    line = std::regex_replace(line, std::regex("^ +"), "");

    if (line.size() > 0) {
      if (line.rfind("[model]") != std::string::npos) {
        model_section = true;
      } else if (line.size() > 0 && line.rfind("[") != std::string::npos) {
        model_section = false;
      }
    }

    if (model_section) {
      if (line.size() > 0 && line.rfind("model_precision") != std::string::npos) {
        line = std::regex_replace(line, std::regex("^ +"), "");
        if (line.rfind("fp16") != std::string::npos) {
          return true;
        }
      }
    }
  }
  return false;
}

void convert_meta_info(fs::path from_model_meta, fs::path to_model_meta) {
  std::string line;

  bool equivalent = false;
  fs::path out_model_meta;
  fs::path tmpPath;
  if (fs::equivalent(from_model_meta, to_model_meta)) {
      equivalent = true;
      out_model_meta = fs::temp_directory_path() / "model_meta";
    }
  else {
    out_model_meta = to_model_meta;
  }

  io::InputFileStream in(from_model_meta);
  io::OutputFileStream out(out_model_meta);

  bool model_section=false;
  while(getline(in, line)) {
    line = std::regex_replace(line, std::regex("^ +"), "");

    if (line.size() > 0) {
      if (line.rfind("[model]") != std::string::npos) {
        model_section = true;
      } else if (line.size() > 0 && line.rfind("[") != std::string::npos) {
        model_section = false;
      }
    }

    if (model_section) {
      if(line.size() > 0 && line.rfind("model_precision") != std::string::npos) {
        line = std::regex_replace(line, std::regex("^ +"), "");
        out << "model_precision = fp16" << std::endl;
      } else {
        out << line << std::endl;
      }
    } else {
      out << line << std::endl;
    }
  }

  if (!model_section) {
    out << std::endl << "[model]" << std::endl;
    out << "model_precision = fp16" << std::endl;
  }

  if (equivalent) {
    fs::copy_file(out_model_meta, to_model_meta, fs::copy_options::overwrite_existing);
    fs::remove(out_model_meta);
  }
}

void convert_model(fs::path fromPath, fs::path toPath) {

  auto tmpPath = fs::temp_directory_path();
  std::vector<std::string> models = std::vector<std::string>({"model.bin", "model.optimizer.bin"});
  for(auto model : models) {
    fs::path model_bin_fp32 = fromPath / model;
    fs::path model_bin_fp16 = toPath / model;
    fs::path model_npz_fp32 = toPath / ( "_fp32" + std::to_string(std::rand()) + ".npz");

    ABORT_IF(!fs::exists(model_bin_fp32), "fp32 bin model ({}) does not exists", model_bin_fp32.string());

    LOG(info, "Loading fp32 items from bin model");
    std::vector<io::Item> items_fp32 = io::loadItems(model_bin_fp32.string());

    LOG(info, "Removing fp32 bin model");
    fs::remove(model_bin_fp32);

    LOG(info, "Saving fp32 items into npz model");
    io::saveItems(model_npz_fp32.string(), items_fp32);

    LOG(info, "Loading fp32 items from npz model");
    std::vector<io::Item> items_fp16 = io::loadItems(model_npz_fp32.string());

    LOG(info, "Removing fp32 npz model");
    fs::remove(model_npz_fp32);

    LOG(info, "Converting npz items from fp32 into fp16");
    io::convertItems(items_fp16, "float16");

    LOG(info, "Saving fp16 items into bin model");
    io::saveItems(model_bin_fp16.string(), items_fp16);
  }
}

int main(int argc, char** argv) {
  using namespace marian;
  createLoggers();

  auto options = New<Options>();
  {
    YAML::Node config; // @TODO: get rid of YAML::Node here entirely to avoid the pattern. Currently not fixing as it requires more changes to the Options object.
    auto cli = New<cli::CLIWrapper>(
        config,
        "Convert a model from fp32 to fp16",
        "Examples:\n"
        "  ./marian-tofp16 -f model.bin -t model.bin  # to store the new model onto e new directory\n"
        "  ./marian-tofp16 -f model.bin --overwrite   # to overwriting original model\n");
    cli->add<std::string>("--from,-f", "Path to dir containing the model to convert.");
    cli->add<std::string>("--to,-t", "Path to directory where to store the new model.");
    cli->add<bool>("--overwrite", "Overwrite the new model onto the original model. Default is false.");
    cli->parse(argc, argv);
    options->merge(config);
  }

  bool overwrite = options->get<bool>("overwrite");

  fs::path fromPath = fs::path(options->get<std::string>("from"));
  ABORT_IF(!fs::exists(fromPath) || fs::is_empty(fromPath) ,
           "Specified input directory {} does not exist, or it is empty",
           fromPath.string());

  fs::path toPath;
  if (!overwrite) {
    toPath = fs::path(options->get<std::string>("to"));
    ABORT_IF(fs::exists(toPath),
             "Specified output directory {} already exists; please remote it.",
             toPath.string());
    fs::create_directories(toPath);
  }

  if (is_fp16(fromPath / "model.meta")) {
    LOG(info, "The model is already fp16; just copy.");
    LOG(info, "Copy input model directory ({}) into output model directory ({})",
        fromPath.string(), toPath.string());

    if (!overwrite) {
      std::filesystem::copy(
          fromPath.string(), toPath.string(), std::filesystem::copy_options::recursive);
    }
  } else {
    LOG(info, "The model is fp32; start conversion.");
    auto tmpPath = overwrite ? fs::temp_directory_path() : fromPath;

    if (overwrite) {
      convert_meta_info(fromPath / "model.meta", fromPath / "model.meta");
      convert_model(fromPath, fromPath);
    } else {
      std::filesystem::copy(
          fromPath.string(), toPath.string(), std::filesystem::copy_options::recursive);
      convert_meta_info(toPath / "model.meta", toPath / "model.meta");
      convert_model(toPath, toPath);
    }
  }
  LOG(info, "Finished");

  return 0;
}
