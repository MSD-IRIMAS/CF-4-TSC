window.HELP_IMPROVE_VIDEOJS = false;

var IMAGES_BASE = "./static/images/";
var NUM_IMG_HFCN = 4;
var NUM_IMG_HINCEPTION = 3;
var NAME_SOTA_HFCN = ['FCN', 'ResNet', 'Inception', 'InceptionTime'];
var NAME_SOTA_HINCEPTION  = ['Inception', 'InceptionTime', 'ROCKET'];

var interp_images_hfcn = [];
var interp_images_hinception = [];


function preloadImages() {
  for (var i = 0; i < NUM_IMG_HFCN; i++) {
    var path_hfcn = IMAGES_BASE + '/results_h-fcn' + String(i+1) + '.png';
    interp_images_hfcn[i] = new Image();
    interp_images_hfcn[i].src = path_hfcn;
  }
  for (var i = 0; i < NUM_IMG_HINCEPTION; i++) {
    var path_hinception = IMAGES_BASE + '/results_h-inception' + String(i+1) + '.png';
    interp_images_hinception[i] = new Image();
    interp_images_hinception[i].src = path_hinception;
  }
}

function setHFCNImage(i) {
  var image = interp_images_hfcn[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#hfcn_sota-image-wrapper').empty().append(image);

  $('#name-sota-hfcn').empty().append(NAME_SOTA_HFCN[i]);
}

function setHINCEPTIONImage(i) {
  var image = interp_images_hinception[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#hinception_sota-image-wrapper').empty().append(image);

  $('#name-sota-hinception').empty().append(NAME_SOTA_HINCEPTION[i]);
}

$(document).ready(function() {
    
    preloadImages();

    $('#hfcn-slider').on('input', function(event) {
      setHFCNImage(this.value);
    });
    setHFCNImage(0);
    $('#hfcn-slider').prop('max', NUM_IMG_HFCN - 1);

    $('#hinception-slider').on('input', function(event) {
      setHINCEPTIONImage(this.value);
    });
    setHINCEPTIONImage(0);
    $('#hinception-slider').prop('max', NUM_IMG_HINCEPTION - 1);

    bulmaSlider.attach();

})
