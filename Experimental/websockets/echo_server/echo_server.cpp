#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>

#include <iostream>
#include <fstream>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;
using namespace std;

// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;

// Define a callback to handle incoming messages
void on_message(server* s, const vector<float*> data, int numRows, websocketpp::connection_hdl hdl, message_ptr msg) {
  std::cout << "on_message called with hdl: " << hdl.lock().get() << " and message: " << msg->get_payload()
            << std::endl;
  string strMsg(msg->get_payload());

  try {
    size_t semiColonPos = strMsg.find(';');
    if (semiColonPos == string::npos)
      throw 1;
    int msgId = boost::lexical_cast<int>(strMsg.substr(0, semiColonPos));
    cout << msgId << endl;
    int numCols = data.size();
    size_t bufferSize = numRows * numCols * 4 + 12;
    char* outBuffer = new char[bufferSize];
    memcpy(outBuffer, static_cast<void*>(&msgId), 4);
    memcpy(outBuffer + 4, static_cast<void*>(&numRows), 4);
    memcpy(outBuffer + 8, static_cast<void*>(&numCols), 4);
    for (int c = 0; c < numCols; ++c) {
      for (int r = 0; r < numRows; ++r) {
        memcpy(outBuffer + 12 + numRows * c * 4 + r * 4, static_cast<void*>(&(data[c][r])), 4);
      }
    }

    // s->send(hdl, msg->get_payload(), msg->get_opcode());
    s->send(hdl, outBuffer, bufferSize, websocketpp::frame::opcode::binary);
  } catch (const websocketpp::lib::error_code& e) {
    std::cout << "Echo failed because: " << e << "(" << e.message() << ")" << std::endl;
  }
}

int main() {
  // Create a server endpoint
  server echo_server;

  try {
    ifstream inFiles[6];
    inFiles[0].open("binary/goog_x.data", ios::in | ios::binary | ios::ate);
    inFiles[1].open("binary/goog_y.data", ios::in | ios::binary | ios::ate);
    inFiles[2].open("binary/time.data", ios::in | ios::binary | ios::ate);
    inFiles[3].open("binary/donation_amount.data", ios::in | ios::binary | ios::ate);
    inFiles[4].open("binary/party.data", ios::in | ios::binary | ios::ate);
    inFiles[5].open("binary/seat.data", ios::in | ios::binary | ios::ate);
    if (inFiles[0].is_open())
      cout << "file 0 is open" << endl;
    vector<float*> data(6);
    streampos size;
    for (int f = 0; f < 6; ++f) {
      size = inFiles[f].tellg();
      cout << size << endl;
      char* memblock = new char[size];
      inFiles[f].seekg(0, ios::beg);
      inFiles[f].read(memblock, size);
      inFiles[f].close();
      data[f] = (float*)memblock;
      cout << data[f][0] << endl;
    }

    // Set logging settings
    echo_server.set_access_channels(websocketpp::log::alevel::all);
    echo_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

    // Initialize ASIO
    echo_server.init_asio();

    // Register our message handler
    echo_server.set_message_handler(bind(&on_message, &echo_server, boost::ref(data), size / 4, ::_1, ::_2));

    // Listen on port 9002
    echo_server.listen(9002);
    cout << "Starting server" << endl;

    // Start the server accept loop
    echo_server.start_accept();

    // Start the ASIO io_service run loop
    echo_server.run();
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  } catch (websocketpp::lib::error_code e) {
    std::cout << e.message() << std::endl;
  } catch (...) {
    std::cout << "other exception" << std::endl;
  }
}
