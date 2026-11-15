import {Composition} from 'remotion';
import {Demo, demoSchema} from './Demo';

export const FPS = 30;

export const Root = () => (
  <Composition id="Demo" component={Demo} durationInFrames={894} fps={FPS} width={1920} height={1080}
    defaultProps={demoSchema} />
);
