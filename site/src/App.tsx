import DevGallery from './DevGallery';
import TopBar from './components/TopBar';
import Hero from './sections/Hero';
import Gallery from './sections/Gallery';
import Compare from './sections/Compare';
import HowItWorks from './sections/HowItWorks';
import Viewer from './sections/Viewer';
import Limits from './sections/Limits';
import Team from './sections/Team';
import FinalCta from './sections/FinalCta';

export default function App() {
  if (location.search.includes('dev=1')) {
    return <DevGallery />;
  }
  return (
    <>
      <TopBar />
      <main>
        <Hero />
        <Gallery />
        <Compare />
        <HowItWorks />
        <Viewer />
        <Limits />
        <Team />
        <FinalCta />
      </main>
    </>
  );
}
