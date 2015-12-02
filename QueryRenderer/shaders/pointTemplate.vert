// VERTEX SHADER

#version 410 core

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

#define useKey <useKey>


in inTx x;
in inTy y;
// in inTz z;

in inTsize size;
in inTfillColor fillColor;

// in vec4 color;

in uint id;


#if useKey==1
// TODO(croot): use the NV_vertex_attrib_integer_64bit extension to use an int64 for the key and invalidKey.
in int key;
uniform int invalidKey;
#endif


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

    #if useKey==1
    if (key != invalidKey) {
        gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);
        gl_PointSize = float(getsize(size));

        fColor = getfillColor(fillColor);
    }
    else {
        gl_Position = vec4(0,0,0,0);
        gl_PointSize = 0.0;
        fColor = vec4(0,0,0,0);
    }
    #else
    gl_Position = vec4(float(getx(x)), float(gety(y)), 0.5, 1.0);
    gl_PointSize = float(getsize(size));

    fColor = getfillColor(fillColor);
    #endif

    // ids from queries go from 0 to numrows-1, but since we're storing
    // the ids as unsigned ints, and there isn't a way to specify the
    // clear value for secondary buffers, we need to account for that
    // offset here
    fPrimitiveId = id + 1;
}
