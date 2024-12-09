import React from 'react';

function Footer() {
  return (
    <footer>
      <div className="firstpanel">Back to top</div>
      <div className="secondpanel">
        <ul>
          <p>Get to Know Us</p>
          <a href="#">Careers</a>
          <a href="#">Blog</a>
          <a href="#">About Amazon</a>
        </ul>
        {/* More columns as in your HTML code */}
      </div>
      <div className="thirdpanel">
        <div className="logo"><img src="/images/logo.png" alt="Amazon Logo" /></div>
      </div>
      <div className="panel4">
        <p>&copy; 1996-2023, Amazon.com, Inc. or its affiliates</p>
      </div>
    </footer>
  );
}

export default Footer;
