// VERTEX SHADER

#version 410 core

#define inTkey <inTkeyType>
#define outTkey <outTkeyType>

#define inTx <inTxType>
#define outTx <outTxType>

#define inTy <inTyType>
#define outTy <outTyType>

// #define inTz float
// #define outTz float

#define inTsize <inTsizeType>
#define outTsize <outTsizeType>

#define inTfillColor <inTfillColorType>
#define outTfillColor <outTfillColorType>



in inTkey key;
in inTx x;
in inTy y;
// in inTz z;

in inTsize size;
in inTfillColor fillColor;

// in vec4 color;

in uint id;


uniform int invalidKey;
uniform int useKey;


outTx getx(in inTx x) {
    return x;
}

outTy gety(in inTy y) {
    return y;
}

// outTz getz(in inTz z) {
//     return z;
// }

outTsize getsize(in inTsize size) {
    return size;
}

outTfillColor getfillColor(in inTfillColor fillColor) {
    return fillColor;
}


////////////////////////////////////////////////////////////////
/**
 * Non-interpolated shader outputs.
 */
flat out uint fPrimitiveId;  // the id of the primitive
flat out vec4 fColor;        // the output color of the primitive

void main() {

    if (useKey == 0 || int(key) != invalidKey) {
        gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);
        gl_PointSize = float(getsize(size));

        fColor = getfillColor(fillColor);
    }
    else {
        gl_Position = vec4(0,0,0,0);
        gl_PointSize = 0.0;
        fColor = vec4(0,0,0,0);
    }
    fPrimitiveId = id;
}
