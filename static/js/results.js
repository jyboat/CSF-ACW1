$(document).ready(function () {
    // Get audio file paths from hidden div
    const $audioData = $("#audio-data");
    const coverPath = $audioData.data("cover");
    const stegoPath = $audioData.data("stego"); 

    //Cover audio
    const wavesurferCover = WaveSurfer.create({
        container: "#waveform-cover",
        waveColor: '#007bff',
        progressColor: '#0056b3',
        height: 80
    });
    wavesurferCover.load(coverPath);

    $('#playCover').on("click", function () {
        wavesurferCover.playPause(); 
    });
 

    //Stego audio
    const wavesurferStego = WaveSurfer.create({
        container: "#waveform-stego",
        waveColor: '#007bff',
        progressColor: '#0056b3',
        height: 80
    });
    wavesurferStego.load(stegoPath);

    $("#playStego").on("click", function () {
        wavesurferStego.playPause();
    });
});

