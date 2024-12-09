import React, { useState } from 'react';

function Header() {
  const [searchTerm, setSearchTerm] = useState('');

  return (
    <header className="navbar">
      <div className="navlogo"><img src="/images/logo.png" alt="Amazon Logo" /></div>
      <div className="nav-search">
        <select className="search-option1">
          <option>All</option>
        </select>
        <input
          type="text"
          placeholder="Search Amazon"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        <button className="search-icon"><i className="fa fa-search"></i></button>
      </div>
      <div className="cart border">
        <i className="fa fa-shopping-cart"></i> Cart
      </div>
    </header>
  );
}

export default Header;
