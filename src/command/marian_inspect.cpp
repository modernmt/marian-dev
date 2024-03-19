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
        "Inspect a model",
        "Examples:\n"
        "  ./marian-conv -m model.bin -t model.npz");
    cli->add<std::string>("--model,-m", "model to inspect", "model.bin");
    cli->add<std::string>("--parameter,-p", "parameter to inspect", "Wemb");
    cli->parse(argc, argv);
    options->merge(config);
  }

  auto modelFrom = options->get<std::string>("model");
  auto parameterName = options->get<std::string>("parameter");

  LOG(info, "model loaded from {}", modelFrom);
  std::vector<io::Item> items = io::loadItems(modelFrom);

  LOG(info, "Finished");

  return 0;
}
