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
        "Convert a model from fp32 to fp16",
        "Examples:\n"
        "  ./marian-tofp16 -f model.bin -t model.bin");
    cli->add<std::string>("--from,-f", "Input path", "model");
    cli->add<std::string>("--to,-t", "Output path", "model");
    cli->parse(argc, argv);
    options->merge(config);
  }

  auto modelFromPath = options->get<std::string>("from");
  auto modelToPath = options->get<std::string>("to");

  auto model_bin_fp32 = modelFromPath + "/model.bin";
  auto model_npz_fp32 = modelToPath + "/npz_fp32_model.npz";
  auto model_npz_fp16 = modelToPath + "/npz_fp16_model.npz";
  auto model_bin_fp16 = modelToPath + "/model.bin";

  LOG(info, "loading fp32 items from bin model ({})", model_bin_fp32);
  std::vector<io::Item> items_fp32 = io::loadItems(model_bin_fp32);

  LOG(info, "saving fp32 items into npz model ({})", model_npz_fp32);
  io::saveItems(model_npz_fp32, items_fp32);

  LOG(info, "\"loading fp32 items from npz model ({})", model_npz_fp32);
  std::vector<io::Item> items_fp16 = io::loadItems(model_npz_fp32);

  LOG(info, "converting npz items from fp32 into fp16");
  io::convertItems(items_fp16, "float16");

  LOG(info, "saving fp16 items into npz model ({})", model_npz_fp16);
  io::saveItems(model_npz_fp16, items_fp16);

  LOG(info, "saving fp16 items into bin model ({})", model_bin_fp16);
  io::saveItems(model_bin_fp16, items_fp16);

  LOG(info, "Finished");

  return 0;
}
