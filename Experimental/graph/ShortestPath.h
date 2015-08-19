#ifndef _SHORTEST_PATH_H
#define _SHORTEST_PATH_H

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
using namespace std;

enum Status { SUCCESS, NODE_NOT_FOUND, NO_PATH };

struct Edge {
  unsigned int fromNodeId;
  unsigned int destNodeId;
};

#define MAX_UNSIGNED 4294967295

template <typename T>
class ShortestPath {
 public:
  ShortestPath() : numNodes_(0), numEdges_(0) {}

  void readEdgesFromFile(const string& fileName) {
    ifstream inFile(fileName.c_str());
    if (inFile.is_open()) {
      T fromNode;
      T destNode;
      while (inFile) {
        inFile >> fromNode >> destNode;
        Edge edge;
        edge.fromNodeId = createNodeDictId(fromNode);
        edge.destNodeId = createNodeDictId(destNode);
        edges_.push_back(edge);
      }
    }
    inFile.close();
    numEdges_ = edges_.size();
    cout << "Num nodes: " << numNodes_ << endl;
  }

  Status findShortestPath(const T& fromNode, const T& destNode, vector<T>& nodePath) {
    unsigned int fromNodeId, destNodeId;
    if (getNodeDictId(fromNode, fromNodeId) == false)
      return NODE_NOT_FOUND;
    if (getNodeDictId(destNode, destNodeId) == false)
      return NODE_NOT_FOUND;
    vector<unsigned int> nodes(numNodes_, MAX_UNSIGNED);
    // seed fromNode
    nodes[fromNodeId] = fromNodeId;

    int round = 1;
    unsigned int edgeNulls = 0;
    unsigned int edgeTerminations = 0;
    while (nodes[destNodeId] == MAX_UNSIGNED && edgeNulls + edgeTerminations < numEdges_) {
      edgeNulls = 0;
      edgeTerminations = 0;
      for (unsigned int e = 0; e != numEdges_; ++e) {
        if (nodes[edges_[e].fromNodeId] == MAX_UNSIGNED)
          edgeNulls++;
        else {
          if (nodes[edges_[e].destNodeId] != MAX_UNSIGNED)
            edgeTerminations++;
          else
            nodes[edges_[e].destNodeId] = edges_[e].fromNodeId;
        }
      }
      cout << "Round: " << round << endl;
      cout << "Edge nulls: " << edgeNulls << endl;
      cout << "Edge Terminations: " << edgeTerminations << endl;
      cout << "Leftover edges: " << numEdges_ - edgeNulls - edgeTerminations << endl;
      round++;
    }
    if (nodes[destNodeId] == MAX_UNSIGNED)
      return NO_PATH;

    nodePath.resize(round);
    nodePath[round - 1] = getNodeForId(destNodeId);
    unsigned int lastNode = destNodeId;
    for (int r = round - 2; r >= 0; --r) {
      nodePath[r] = getNodeForId(nodes[lastNode]);
      lastNode = nodes[lastNode];
    }
    return SUCCESS;
  }

  bool getNodeDictId(const T& node, unsigned int& nodeId) {
    auto idIt = dict_.find(node);
    if (idIt != dict_.end()) {
      nodeId = idIt->second;
      return true;
    }
    return false;
  }

 private:
  std::map<T, unsigned int> dict_;
  unsigned int numNodes_;
  unsigned int numEdges_;
  vector<T> dictKeys_;
  vector<Edge> edges_;

  unsigned int createNodeDictId(const T& node) {
    auto idIt = dict_.find(node);
    if (idIt != dict_.end())
      return idIt->second;
    dict_[node] = numNodes_++;
    dictKeys_.push_back(node);
    return numNodes_ - 1;
  }

  inline T getNodeForId(const unsigned int nodeId) { return dictKeys_[nodeId]; }
};

#endif
