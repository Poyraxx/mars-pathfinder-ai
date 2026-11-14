// Video authoring uses the exact pure geometry module rendered by React.
import {navigationRibbon} from '../src/routeGeometry.ts';

let input='';
for await (const chunk of process.stdin)input+=chunk.toString('utf8');
const frames=JSON.parse(input);
if(!Array.isArray(frames))throw new Error('Expected an array of route geometry inputs');
process.stdout.write(JSON.stringify(frames.map(frame=>navigationRibbon(
 frame.ribbons??[],frame.groundM,frame.segments??[],frame.localLength??0,frame.planningLink??false,frame.manoeuvre??false
))));
