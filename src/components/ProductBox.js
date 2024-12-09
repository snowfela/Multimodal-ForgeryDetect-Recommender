import React from 'react';

function ProductBox({ title, image, link }) {
  return (
    <div className="box">
      <h2>{title}</h2>
      <div className="box1-img" style={{ backgroundImage: `url(${image})` }}></div>
      <p><a href={link}>Shop now</a></p>
    </div>
  );
}

export default ProductBox;
