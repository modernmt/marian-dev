#include "marian.h"
#include "common/binary.h"

using namespace marian;

int main(int argc, char** argv) {
  using namespace marian;
  createLoggers();

  auto options = New<Options>();
  {
    YAML::Node config; // @TODO: get rid of YAML::Node here entirely to avoid the pattern. Currently not fixing as it requires more changes to the Options object.
    auto cli = New<cli::CLIWrapper>(
        config,
        "Convert a model from bin to npz",
        "Examples:\n"
        "  ./marian-conv -f model.bin -t model.npz");
    cli->add<std::string>("--from,-f", "Input path", "model");
    cli->add<std::string>("--to,-t", "Output apth", "model");
    cli->parse(argc, argv);
    options->merge(config);
  }

  auto modelFromPath = options->get<std::string>("from");
  auto modelToPath = options->get<std::string>("to");

  auto model_bin_fp32 = modelFromPath + "/model.bin";
  auto model_npz_fp32 = modelToPath + "/npz_fp32_model.npz";
  auto model_npz_fp16 = modelToPath + "/npz_fp16_model.npz";
  auto model_bin_fp16 = modelToPath + "/bin_fp16_model.bin";

  LOG(info, "items_fp32 bin loading from {}", model_bin_fp32);
  std::vector<io::Item> items_fp32 = io::loadItems(model_bin_fp32);

  LOG(info, "items_fp32 npz saving into {}", model_npz_fp32);
  io::saveItems(model_npz_fp32, items_fp32);

  LOG(info, "items_fp16 npz loading from {}", model_npz_fp32);
  std::vector<io::Item> items_fp16 = io::loadItems(model_npz_fp32);

  LOG(info, "items_fp16 converting into {}", "float16");
  io::convertItems(items_fp16, "float16");

  LOG(info, "items_fp16 npz saving into {}", model_npz_fp16);
  io::saveItems(model_npz_fp16, items_fp16);

  LOG(info, "items_fp16 bin saving into {}", model_bin_fp16);
  io::saveItems(model_bin_fp16, items_fp16);

  LOG(info, "Finished");

  return 0;
}
