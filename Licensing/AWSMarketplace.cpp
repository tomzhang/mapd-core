#include "cpr/cpr.h"
#include "rapidjson/document.h"
#include "Licensing/AWSBillingProducts.h"
#include "QueryEngine/JsonAccessors.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <iostream>
#include <vector>
#include <string>

bool is_valid_billingproduct(const rapidjson::Document& doc, const std::vector<std::string>& validBillingProducts) {
  // Assume valid if no billing products provided
  if (validBillingProducts.size() == 0 || (validBillingProducts.size() == 1 && validBillingProducts[0] == "")) {
    return true;
  }
  const auto& billingProducts = field(doc, "billingProducts");
  if (!billingProducts.IsArray()) {
    throw std::runtime_error("AWS: billingProducts from instance metadata should be an array");
  }
  for (auto bp = billingProducts.Begin(); bp != billingProducts.End(); ++bp) {
    for (auto validBillingProduct : validBillingProducts) {
      if (validBillingProduct == bp->GetString()) {
        return true;
      }
    }
  }
  return false;
}

rapidjson::Document download_instance_metadata() {
  // auto d = cpr::Get(cpr::Url{"http://andrew-testing.frontend.builds.mapd.com/aws/rhel/document"});
  auto d = cpr::Get(cpr::Url{"http://169.254.169.254/latest/dynamic/instance-identity/document"}, cpr::Timeout{5000});
  if (d.status_code != 200) {
    if (d.error.code == cpr::ErrorCode::OPERATION_TIMEDOUT) {
      throw std::runtime_error("AWS: could not contact instance metadata server");
    } else {
      throw std::runtime_error("AWS: failed to download instance metadata");
    }
  }
  rapidjson::Document metadata;
  metadata.Parse(d.text.c_str());
  return metadata;
}

bool validate_server() {
  std::vector<std::string> billingProducts;
  boost::split(billingProducts, AWS_BILLINGPRODUCTS, boost::is_any_of(","));
  auto metadata = download_instance_metadata();
  return is_valid_billingproduct(metadata, billingProducts);
}
