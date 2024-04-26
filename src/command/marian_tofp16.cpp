#include <filesystem>
#include "marian.h"
#include "common/binary.h"

using namespace marian;
namespace fs = std::filesystem;

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
        "  ./marian-tofp16 -f model.bin -t model.bin");
    cli->add<std::string>("--from,-f", "Path to dir containing the model to convert");
    cli->add<std::string>("--to,-t", "Path to directory where to store the new model.");
    cli->parse(argc, argv);
    options->merge(config);
  }

  fs::path fromPath = fs::path(options->get<std::string>("from"));
  fs::path toPath = fs::path(options->get<std::string>("to"));

  ABORT_IF(!fs::exists(fromPath) || fs::is_empty(fromPath) ,
           "Specified input directory {} does not exist, or it is empty",
           fromPath.string());
  ABORT_IF(fs::exists(toPath),
           "Specified output directory {} already exists; please remote it.",
           toPath.string());

  auto tmpPath = fs::temp_directory_path();
  std::vector<std::string> models = std::vector<std::string>({"model.bin", "model.optimizer.bin"});


  LOG(info,
      "Copy input model directory ({}) into output model directory ({})",
      fromPath.string(),
      toPath.string());
  std::filesystem::copy(
      fromPath.string(), toPath.string(), std::filesystem::copy_options::recursive);

  for (auto model : models) {

    fs::path model_bin_fp32 = fromPath / model;
    fs::path model_bin_fp16 = toPath / model;
    fs::path model_npz_fp32 = tmpPath / "model_fp32.npz";
    fs::path model_npz_fp16 = tmpPath / "model_fp16.npz";

    if (!fs::exists(model_bin_fp32))
      continue;

    LOG(info,
        "remove binary model ({}) from output model directory ({})",
        model_bin_fp32.string(), toPath.string());
    fs::remove(model_bin_fp16);

    LOG(info, "Loading fp32 items from bin model ({})", model_bin_fp32.string());
    std::vector<io::Item> items_fp32 = io::loadItems(model_bin_fp32.string());

    LOG(info, "Saving fp32 items into npz model ({})", model_npz_fp32.string());
    io::saveItems(model_npz_fp32.string(), items_fp32);

    LOG(info, "Loading fp32 items from npz model ({})", model_npz_fp32.string());
    std::vector<io::Item> items_fp16 = io::loadItems(model_npz_fp32.string());

    LOG(info, "Converting npz items from fp32 into fp16");
    io::convertItems(items_fp16, "float16");

    LOG(info, "Saving fp16 items into npz model ({})", model_npz_fp16.string());
    io::saveItems(model_npz_fp16.string(), items_fp16);

    LOG(info, "Saving fp16 items into bin model ({})", model_bin_fp16.string());
    io::saveItems(model_bin_fp16.string(), items_fp16);

    fs::remove(model_npz_fp32);
    fs::remove(model_npz_fp16);
  }
  LOG(info, "Finished");

  return 0;
}
