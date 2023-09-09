document.addEventListener("DOMContentLoaded", function () {
  // Get the file input and the label text element
  const fileInput = document.getElementById("fileInput");
  const fileLabelText = document.getElementById("fileLabelText");
  const colorSwatches = document.querySelectorAll(".color-swatch");

  // Add an event listener to the file input
  fileInput.addEventListener("change", function () {
      // Check if a file is selected
      if (fileInput.files.length > 0) {
          // Update the label text with the selected file's name
          fileLabelText.innerText = fileInput.files[0].name;
      } else {
          // If no file is selected, revert to the default text
          fileLabelText.innerText = "Choose a Photo";
      }
  });

  // Add click event listener to color swatches
  colorSwatches.forEach((swatch) => {
      swatch.addEventListener("click", function () {
          const colorInfo = swatch.querySelector(".color-info");
          const rgbValue = colorInfo.innerText.trim(); // Get the RGB value

          // Create a temporary input element to copy the RGB value to the clipboard
          const tempInput = document.createElement("input");
          tempInput.value = rgbValue;
          document.body.appendChild(tempInput);
          tempInput.select();
          document.execCommand("copy");
          document.body.removeChild(tempInput);

          // Show a message indicating that the color code has been copied
          alert("Copied color code: " + rgbValue);
      });
  });
});
