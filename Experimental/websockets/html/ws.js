var wsMan;

function WebSocketManager (address,port) {
  this.address = address;
  this.port = port;
  this.endpoint = null;
  this.ws = null;
  this.msgId = 0;
  this.callbackMap = {};

  this.onOpen = function() {
    console.log("opened");
  }
  this.onMessage = function(evt) {
    //console.log("Received: " + evt.data);
    this.msgCount++;
    if (this.msgCount % 100 == 0)
      console.log(this.msgCount);
    this.sendMessage(evt.data);
  }
  this.onClose = function() {
    console.log("closed");
  }

  this.onError = function(evt) {
    console.log("Error: " + evt.data);
  }

  this.init = function() {
    if (!("WebSocket" in window))
      return false;
    this.endpoint = "ws://" + this.address + ":" + this.port + "/echo";
    this.ws = new WebSocket(this.endpoint);
    this.ws.onerror = this.onError;
    this.ws.onopen = this.onOpen;
    this.ws.onmessage = $.proxy(this.onMessage,this);
    this.ws.onclose = this.onClose;
    return true;
  }

  this.isGood = function() {
    return this.ws.readyState === 1; 
  }

  this.sendMessage = function(msg, callback) {
    var curMsgId = this.msgId++;
    msg = curMsgId + ";" + msg;
    this.callbackMap[curMsgId] = callback; 

    //console.log("Sending: " + msg); 
    this.ws.send(msg);
    //console.log("sent");
  }

}

$(document).ready(function() {
  wsMan = new WebSocketManager("localhost","9002");
  var status = wsMan.init();
  console.log("Status: " + status);
  $("#sendRequest").click($.proxy(wsMan.sendMessage,wsMan,"yaya"));
});


