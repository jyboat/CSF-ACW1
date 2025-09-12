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

    const $textPayload = $("#textPayload");
    const $payloadFile = $("#payloadFile");
    const $hiddenTextPayload = $("#hiddenTextPayload");
    const $hiddenFilePayload = $("#hiddenFilePayload");
    const $removePayloadBtn = $("#removePayloadBtn");
    const $hideBtn = $("#hideBtn");
    const $form = $("#stegoForm");

    const $lsbCount = $("#lsbCount");
    const $capacityInfo = $("#capacityInfo");

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
        checkCapacity();
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
        $capacityInfo.text("Waiting for payload input...");
    });

    $payloadFile.on("change", function (e) {
        const file = e.target.files[0];
        if (!file) return;
        $removePayloadBtn.removeClass("d-none");
        checkCapacity();
    });

    $removePayloadBtn.on("click", function () {
        $payloadFile.val("");
        $removePayloadBtn.addClass("d-none");
        checkCapacity();
    });

    $textPayload.on("input", checkCapacity);
    $lsbCount.on("change", checkCapacity);

    $previewBox.on("click", function (e) {
        // Only proceed if a file is uploaded
        if ($fileInput[0].files.length === 0) {
            $fileInput.trigger("click");
            return; // no file, do nothing
        }

        // Calculate where user clicked
        const offset = $(this).offset();
        const x = e.pageX - offset.left;
        const y = e.pageY - offset.top;
        console.log("Clicked at:", x, y);

        // Show Bootstrap modal
        const locationModal = new bootstrap.Modal(document.getElementById('locationModal'));
        locationModal.show();
    });

    // Drag and drop stuff
    $previewBox.on("dragover dragenter", function(e) {
        e.preventDefault();
        e.stopPropagation();
        $previewBox.css("background-color", "#f3e8ff");
    });

    $previewBox.on("dragleave dragend drop", function(e) {
        e.preventDefault();
        e.stopPropagation();
        $previewBox.css("background-color", "");
    });

    // Handle file drop
    $previewBox.on("drop", function(e) {
        e.preventDefault();
        e.stopPropagation();
        $previewBox.removeClass("drag-over"); // remove highlight

        const files = e.originalEvent.dataTransfer.files;
        if (!files || files.length === 0) return;

        const file = files[0]; // only handle the first file for now
        const allowedTypes = ["image/", "audio/"];
        const allowedExtensions = [".bmp", ".png", ".gif", ".wav", ".pcm"];

        // Check MIME type
        const isAllowedType = allowedTypes.some(type => file.type.startsWith(type));
        // Or check extension as fallback
        const isAllowedExt = allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

        if (!isAllowedType && !isAllowedExt) {
            toastr.error("Unsupported file type! Please upload an image (.bmp, .png, .gif) or audio (.wav, .pcm).");
            return;
        }

        // Assign file to input so form submission works
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        $fileInput[0].files = dataTransfer.files;

        // Trigger change event
        $fileInput.trigger("change");
    });

    // Show or hide step 3 text
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

    $hideBtn.on("click", function () {
        const textVal = $textPayload.val().trim();
        const fileVal = $payloadFile[0].files.length > 0;

        if (textVal && fileVal) {
            toastr.error("Please choose either text input OR a file, not both.");
            return;
        }

        if (!textVal && !fileVal) {
            toastr.error("Missing fields, please enter text OR upload a file as your payload.");
            return;
        }

        if (!checkCapacity(true)) {
            toastr.error("Insufficient cover capacity for this payload. Please choose fewer payload bits, reduce payload size, or increase LSBs.");
            return;
        }

        // Clear previous hidden values
        $hiddenTextPayload.val("");
        $hiddenFilePayload.val("");

        if (textVal) {
            // Pass text to hidden field
            $hiddenTextPayload.val(textVal);
        } else if (fileVal) {
            // Copy file to hidden file input
            const file = $payloadFile[0].files[0];

            // Trick: assign the file to the hidden input using DataTransfer
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            $hiddenFilePayload[0].files = dataTransfer.files;
        }

        // Submit the main form
        $form.submit();
    });

    function checkCapacity(strict = false) {
        const coverFile = $fileInput[0].files[0];
        const lsbCount = parseInt($lsbCount.val(), 10) || 0;

        if (!coverFile || lsbCount <= 0) {
            $capacityInfo.text("Waiting for cover and LSB selection...");
            $hideBtn.prop("disabled", true)
            return false;
        }

        const coverCapacityBits = coverFile.size * lsbCount;

        let payloadSizeBytes = 0;
        if ($textPayload.val().trim()) {
            payloadSizeBytes = new Blob([$textPayload.val().trim()]).size;
        } else if ($payloadFile[0].files.length > 0) {
            payloadSizeBytes = $payloadFile[0].files[0].size;
        }

        const payloadBits = payloadSizeBytes * 8;

        if (payloadBits === 0) {
            $capacityInfo.text("Waiting for payload input...");
            $hideBtn.prop("disabled", true)
            return false;
        }

        if (coverCapacityBits >= payloadBits) {
            $capacityInfo
                .text(`Sufficient: Cover capacity ${coverCapacityBits} bits, Payload requires ${payloadBits} bits.`)
                .removeClass("text-danger")
                .addClass("text-success");
            $hideBtn.prop("disabled", false)
            return true;
        } else {
            $capacityInfo
                .text(`Insufficient: Cover capacity ${coverCapacityBits} bits, Payload requires ${payloadBits} bits.
                Please reduce payload size, or increase cover size.`)
                .removeClass("text-success")
                .addClass("text-danger");
            $hideBtn.prop("disabled", true)
            return false;
        }
    }
});