import React from 'react';
import ProductBox from './ProductBox';
import productsData from '../data';

function MainContent() {
  return (
    <main className="main-content">
      <div className="banner">
        <p>You can also shop on Amazon India for local delivery. <a href="#">Go to amazon.in</a></p>
      </div>
      <div className="shop">
        {productsData.map((product) => (
          <ProductBox key={product.id} title={product.title} image={product.image} link={product.link} />
        ))}
      </div>
    </main>
  );
}

export default MainContent;
