import Section from './components/Section';
import Button from './components/Button';
import { Eyebrow, H1, H2 } from './components/Heading';
import VideoLoop from './components/VideoLoop';
import Card from './components/Card';
import CompareSlider from './components/CompareSlider';

function Group({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 'var(--space-6)' }}>
      <p
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          color: 'var(--muted)',
          marginBottom: 'var(--space-3)',
        }}
      >
        {name}
      </p>
      {children}
    </div>
  );
}

export default function DevGallery() {
  return (
    <main>
      <Section id="dev-gallery">
        <Group name="Section (light)">
          <p>This section uses the default light theme.</p>
        </Group>
        <Group name="Button">
          <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
            <Button href="#">Primary</Button>
            <Button href="#" variant="secondary">
              Secondary
            </Button>
          </div>
        </Group>
        <Group name="Heading">
          <Eyebrow>Eyebrow</Eyebrow>
          <H1>Heading one</H1>
          <H2>Heading two</H2>
        </Group>
        <Group name="VideoLoop">
          <div style={{ maxWidth: '480px' }}>
            <VideoLoop src="/media/sample.mp4" poster="/media/sample.jpg" label="Sample loop" />
          </div>
        </Group>
        <Group name="Card">
          <div style={{ maxWidth: '360px' }}>
            <Card
              title="Sample card"
              caption="A caption for the sample card."
              label="sample"
              media={<div style={{ width: '100%', height: '100%', background: '#888' }} />}
            />
          </div>
        </Group>
        <Group name="CompareSlider">
          <CompareSlider
            before={<div style={{ width: '100%', height: '100%', background: '#000' }} />}
            after={<div style={{ width: '100%', height: '100%', background: '#fff' }} />}
            beforeLabel="Source"
            afterLabel="Wander"
          />
        </Group>
      </Section>
      <Section id="dev-gallery-dark" theme="dark">
        <Group name="Section (dark)">
          <p>This section uses the dark theme.</p>
        </Group>
      </Section>
    </main>
  );
}
