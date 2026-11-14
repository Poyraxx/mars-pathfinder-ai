/// <reference types="vite/client" />
declare module 'lerc' {
 const Lerc:{decode:(buffer:ArrayBuffer)=>{width:number;height:number;pixels:Float32Array[];mask?:Uint8Array;maskData?:Uint8Array}};
 export default Lerc;
}
