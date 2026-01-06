Dropzone.autoDiscover = false;

function init() {

    let dz = new Dropzone("#dropzone", {
        url: "/classify_image",          
        maxFiles: 1,
        addRemoveLinks: true,
        autoProcessQueue: false,         
        acceptedFiles: "image/*"         
    });

    
    dz.on("addedfile", function () {
        if (dz.files.length > 1) {
            dz.removeFile(dz.files[0]);
        }
    });

   
    $("#submitBtn").on("click", function () {

        if (dz.files.length === 0) {
            alert("Please upload an image first");
            return;
        }

        let file = dz.files[0];
        let imageData = file.dataURL;

        let url = "http://127.0.0.1:5000/classify_image";

        $.post(url, {
            image_data: imageData
        }, function (data) {

            console.log(data);

            if (!data || data.length === 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;

            for (let i = 0; i < data.length; i++) {
                let maxScore = Math.max(...data[i].class_probability);
                if (maxScore > bestScore) {
                    match = data[i];
                    bestScore = maxScore;
                }
            }

            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();

                $("#resultHolder").html($(`[data-player="${match.class}"]`).html());

                let classDictionary = match.class_dictionary;
                for (let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let probabilityScore = match.class_probability[index];
                    $("#score_" + personName).html(probabilityScore.toFixed(2));
                }
            }
        });
    });
}

$(document).ready(function () {
    console.log("ready!");

    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});
