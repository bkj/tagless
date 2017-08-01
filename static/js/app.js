function next_image(e, label) {  
  $.ajax({
    type: 'POST',
    url: 'label',
    processData: false,
    contentType: 'application/json',
    dataType: 'json',
    data: JSON.stringify({'image_path' : e.target.src, 'label' : label}),
    success: function(responses, status) {
      var urls = $('img').map(function(e, x) {return x.src.split('/').pop()});
      _.map(function(response) {
        if(!_.contains(urls, response.src.split('/').pop())) {
          $('#images').append(
            `<a class="image" style="width: ${response.width}px; height: ${response.height}px">` + 
              `<img src="${response.src}" width="${response.width}" height="${response.height}"/>` +
            `</a>`
          )
          $(".image").unbind('click').unbind('contextmenu')
          $(".image").click(function(e) { 
            $(this).css('background-color', 'green').children().css('opacity', '0.5');
            next_image(e, true) 
          }).contextmenu(function(e) { 
            $(this).css('background-color', 'red').children().css('opacity', '0.5');
            next_image(e, false) 
          });
        }
      });
      $('img').unveil(); 
    }
  });
}

$(document).ready(function() {
    $('img').unveil();
      $(".image").click(function(e) { 
        $(this).css('background-color', 'green').children().css('opacity', '0.5');
        next_image(e, true) 
      }).contextmenu(function(e) { 
        $(this).css('background-color', 'red').children().css('opacity', '0.5');
        next_image(e, false) 
      });
});

