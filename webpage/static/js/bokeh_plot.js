(function() {
var fn = function() {
  Bokeh.safely(function() {
    (function(root) {
      function embed_document(root) {
        
      var docs_json = document.getElementById('1213').textContent;
      var render_items = [{"docid":"d718a390-6239-4aa4-bb81-d28e9d478b86","root_ids":["1005"],"roots":{"1005":"34f8c60f-4258-47e0-a196-e1e134c609d1"}}];
      root.Bokeh.embed.embed_items(docs_json, render_items);
    
      }
      if (root.Bokeh !== undefined) {
        embed_document(root);
      } else {
        var attempts = 0;
        var timer = setInterval(function(root) {
          if (root.Bokeh !== undefined) {
            clearInterval(timer);
            embed_document(root);
          } else {
            attempts++;
            if (attempts > 100) {
              clearInterval(timer);
              console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
            }
          }
        }, 10, root)
      }
    })(window);
  });
};
if (document.readyState != "loading") fn();
else document.addEventListener("DOMContentLoaded", fn);
})();