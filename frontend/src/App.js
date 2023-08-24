import React, { useState } from 'react';
import axios from 'axios';
import "./App.css"

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [pixelateValue, setPixelateValue] = useState(true);
  const [video, setVideo] = useState(false);
  const [methodValue, setMethodValue] = useState('0');

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    setSelectedImage(URL.createObjectURL(file));
    console.log(URL.createObjectURL(file))
    const formData = new FormData();
    formData.append('imagefile', file);
    formData.append('pixelate', pixelateValue);
    formData.append('method', methodValue);
    formData.append('isvideo', video);




    try {
      setProcessedImage(null)
      const response = await axios.post('http://127.0.0.1:5001/blur', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      console.log(response.data)
      setProcessedImage(response.data.processed_image);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="container">
      <h1>Face Blur App</h1>
      <input type="file" onChange={handleImageUpload} />
      <div>
        <label>
          Pixelate:
          <input
            type="checkbox"
            checked={pixelateValue}
            onChange={() => setPixelateValue(!pixelateValue)}
          />
        </label>
        <label>
          Video:
          <input
            type="checkbox"
            checked={video}
            onChange={() => setVideo(!video)}
          />
        </label>
      </div>
      <div>
        <label>
          Method:
          <select
            value={methodValue}
            onChange={(event) => setMethodValue(event.target.value)}
          >
            <option value="0">Method 0</option>
            <option value="1">Method 1</option>
            <option value="2">Method 2</option>
          </select>
        </label>
      </div>
      {(selectedImage && !video) && <img src={selectedImage} alt="Selected" className="selected-image"  width={500}/>}
      {(selectedImage && video) && <video controls  src={selectedImage} alt="Selected" className="selected-image" type="video/mp4" width={500}/>}
      {(processedImage && !video) && <img src={"http://127.0.0.1:5001/"+processedImage.slice(2)} alt="Processed" className="processed-image"  width={500}/>}
      {(processedImage && video) && <video controls  src={"http://127.0.0.1:5001/"+processedImage.slice(2)} alt="Processed" type="video/mp4" className="processed-image"  width={500}/>}
    </div>
  );
};

export default App;
