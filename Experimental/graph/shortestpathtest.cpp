#include "ShortestPath.h"

#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>

using namespace std;

int main() {
  ShortestPath<unsigned int> shortestPath;
  shortestPath.readEdgesFromFile("web-google.txt");

  while (1 == 1) {
    string fromNode;
    string destNode;
    cout << endl << "Enter from_node: ";
    getline(cin, fromNode);
    if (fromNode == "q")
      break;
    cout << "Enter dest_node: ";
    getline(cin, destNode);
    if (destNode == "q")
      break;
    // bool status = shortestPath.getNodeDictId(boost::lexical_cast<unsigned int>(node), nodeId);
    vector<unsigned int> nodePath;
    Status status = shortestPath.findShortestPath(
        boost::lexical_cast<unsigned int>(fromNode), boost::lexical_cast<unsigned int>(destNode), nodePath);

    switch (status) {
      case SUCCESS:
        cout << "Path found" << endl;
        for (auto nodeIt = nodePath.begin(); nodeIt != nodePath.end(); ++nodeIt) {
          cout << *nodeIt << " -> ";
        }
        cout << endl;
        break;
      case NO_PATH:
        cout << "No path between nodes" << endl;
        break;
      case NODE_NOT_FOUND:
        cout << "One or more nodes do not exist" << endl;
        break;
    }
  }
}
