let wavesurfer = null;

$(document).ready(function () {
    const $fileInput = $("#coverFile");
    const $coverImage = $("#coverImage");
    const $waveform = $("#waveform");
    const $fileActions = $("#fileActions");
    const $noFileText = $("#noFileText");
    const $previewBox = $("#previewBox");
    const $stegoKey = $("#stegoKey");
    const $step3Text = $("#step3Text");

    const defaultBoxHeight = 250; // Fixed for audio / empty

    function setBoxHeight(height) {
        $previewBox.css('height', height + 'px');
    }

    setBoxHeight(defaultBoxHeight);

    $fileInput.on("change", function (e) {
        const file = e.target.files[0];
        if (!file) return;

        const fileType = file.type;

        // Reset previews
        $coverImage.addClass("d-none").attr("src", "");
        $waveform.addClass("d-none").empty();
        $noFileText.show();
        if (wavesurfer) {
            wavesurfer.destroy();
            wavesurfer = null;
        }

        if (fileType.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = function (event) {
                $coverImage.attr("src", event.target.result).removeClass("d-none");
                $noFileText.hide();

                // Wait for image to load
                const img = new Image();
                img.src = event.target.result;
                img.onload = function () {
                    const containerWidth = $previewBox.width();
                    const scaleFactor = containerWidth / img.naturalWidth;
                    const scaledHeight = img.naturalHeight * scaleFactor;

                    // Set preview box height to match scaled image
                    setBoxHeight(scaledHeight);
                };
            };
            reader.readAsDataURL(file);

        } else if (fileType.startsWith("audio/") || file.name.endsWith(".wav") || file.name.endsWith(".pcm")) {
            $waveform.removeClass("d-none");
            $noFileText.hide();

            // Fixed height for audio
            setBoxHeight(defaultBoxHeight);

            wavesurfer = WaveSurfer.create({
                container: "#waveform",
                waveColor: '#007bff',
                progressColor: '#0056b3',
                height: defaultBoxHeight
            });
            const audioUrl = URL.createObjectURL(file);
            wavesurfer.load(audioUrl);
        }

        $fileActions.removeClass("d-none");
    });

    $("#removeFileBtn").on("click", function () {
        $fileInput.val("");
        $coverImage.addClass("d-none").attr("src", "");
        $waveform.addClass("d-none").empty();
        $noFileText.show();
        setBoxHeight(defaultBoxHeight);
        if (wavesurfer) {
            wavesurfer.destroy();
            wavesurfer = null;
        }
        $fileActions.addClass("d-none");
        toggleStep3Text();
    });

    function toggleStep3Text() {
        const fileSelected = $fileInput[0].files.length > 0;
        const keyEntered = $stegoKey.val().trim() !== "";

        if (fileSelected && keyEntered) {
            $step3Text.show();
        } else {
            $step3Text.hide();
        }
    }

    // Trigger whenever file changes
    $fileInput.on("change", toggleStep3Text);

    // Trigger whenever key changes
    $stegoKey.on("input", toggleStep3Text);
});