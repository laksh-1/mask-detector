const spawn = require("child_process").spawn;
const express = require("express");
const process = spawn("python3", ["./detect_mask_video.py"]);

const app = express();

app.get("/", function(req, res) {
  res.send("press key 'q' to exit");
});

process.stdout.on("data", data => {
  console.log(data);
});

app.listen(3000, function() {
  console.log("server succesully running on 3000");
});
