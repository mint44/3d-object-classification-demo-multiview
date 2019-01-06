$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            
            reader.readAsText(input.files[0]);
        }
    }
    
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-render').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });
    
    // Render
    $('#btn-render').click(function () {
        console.log('render clicked');
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /render
        $.ajax({
            type: 'POST',
            url: '/render',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log(data);
                data = data.replace(/'/g, '"');
                imgs = JSON.parse(data);
                var container = document.getElementById('rendered-imgs');
                for (var i = 0, j = imgs.length; i < j; i++) {
                    var img = document.createElement('img');
                    img.style.width='120';
                    img.src = imgs[i]; // img[i] refers to the current URL.
                    container.appendChild(img);
                }
            },
        });
    });

    // Predict
    $('#btn-predict').click(function () {
        console.log('predict clicked');
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
                
                
            },
        });
    });

});
