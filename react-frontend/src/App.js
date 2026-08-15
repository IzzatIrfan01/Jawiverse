import React, { useState} from 'react';
import { useVanta } from './hooks/useVanta';
import Header from './components/Header';
import Converter from './components/Converter';
// import Features from './components/Features';
import Footer from './components/Footer';

function App() {
  const [currentLang, setCurrentLang] = useState('en');
  const vantaRef = useVanta();

  const handleLanguageChange = (lang) => {
    setCurrentLang(lang);
  };

  return (
    <div className="bg-[#FDF6EC] min-h-screen flex flex-col relative">
      <div ref={vantaRef} className="vanta-bg"></div>
      
      <Header 
        currentLang={currentLang} 
        onLanguageChange={handleLanguageChange} 
      />
      
      <main className="flex-grow container mx-auto px-4 py-12 relative z-10">
        <Converter currentLang={currentLang} />
        {/* <Features currentLang={currentLang} /> */}
      </main>
      
      <Footer currentLang={currentLang} />
    </div>
  );
}

export default App;