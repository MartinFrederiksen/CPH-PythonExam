window.onload = function() { 
    $("#zip-uploader").submit(function(e) {
        e.preventDefault();
        $("#button-group").empty();
        $("#filter-buttons").empty();
        $("#filter-images").empty();

        var formData = new FormData(this);

        $.ajax({
            url: "/zip/upload",
            type: 'POST',
            data: formData,
            success: function(response) {
                response.staticFolders.ShowAll = []
                for(var key in response.staticFolders) {
                    var btn = document.createElement("BUTTON");
                    btn.innerHTML = key;
                    btn.addEventListener("click", filterSelection, false);
                    btn.setAttribute('data-filter', key);
                    btn.className = "btn btn-light mr-2 mb-2"
                    $("#filter-buttons").prepend(btn)

                    response.staticFolders[key].forEach(imgName => {
                        var img = document.createElement("IMG");
                        img.src = "/static/unzipped/" + response.staticName + "/" + key + "/" + imgName;
                        img.className = "filterImg grid-item " + key;
                        
                        $("#filter-images").append(img)
                    });
                }

                filterSelection()
                $('#filter-images').waitForImages(function() {
                    $('#filter-images').masonry({
                        itemSelector: '.grid-item',
                        columnWidth: 180
                    });
                    $("#filter-images").addClass('show');
                    $("#button-group").append('<form action="/zip/download"><input type="hidden" name="type" value="original" /><input type="hidden" name="zip" value="' + response.originalName + '" /><button class="btn btn-primary">Download Original ZIP</button>')
                    $("#button-group").append('<form class="mr-2" action="/zip/download"><input type="hidden" name="type" value="sorted" /><input type="hidden" name="zip" value="' + response.zippedName + '" /><button class="btn btn-primary">Download Sorted ZIP</button>')
                });
            },
            cache: false,
            contentType: false,
            processData: false
        });
    })
}

function filterSelection() {
    var c = $(this).data('filter');

    var images = $(".filterImg")

    if(c == null || c == "ShowAll")
        c = "";

    for (i = 0; i < images.length; i++) {
        $(images[i]).removeClass('show');
        if (images[i].className.indexOf(c) > -1) 
            $(images[i]).addClass('show')
    }

    $('.grid').masonry({
        itemSelector: '.grid-item',
        columnWidth: 180
    });
}