
/*
    app.js
*/


MAX_IMAGES = 150

function next_image(e, label) {  
  $.ajax({
    type: 'POST',
    url: 'label',
    processData: false,
    contentType: 'application/json',
    dataType: 'json',
    data: JSON.stringify({'image_path' : e.target.src, 'label' : label}),
    success: function(responses, status) {
        _.map(responses, function(response) {
            $('#images').append(
                `<a class="image" style="width: ${response.width}px; height: ${response.height}px">` + 
                    `<img src="${response.src}" width="${response.width}" height="${response.height}"/>` +
                `</a>`
            )
        });
        
        // Don't show more than MAX_IMAGES at a time (too many images seemed to choke the app)
        while($('img').length > ) {
            $('img')[0].remove();
        }
        // $('img').unveil(); 
    }
  });
}

$(document).ready(function() {
    $('img').unveil();
    $("body").on("click", ".image", function(e){
        $(this).css('background-color', 'green').children().css('opacity', '0.5');
        next_image(e, true) 
    });
    $("body").on("contextmenu", ".image", function(e){
        $(this).css('background-color', 'red').children().css('opacity', '0.5');
        next_image(e, false) 
    });
});

