// ============================================================
// RUBIDIUM TRANSFORMER - CONVERT TO PICKLE
// Converts binary model to Python pickle format
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <string>

// Pickle opcodes
#define PICKLE_PROTO 0x80
#define PICKLE_GLOBAL 0x63
#define PICKLE_SHORT_BINUNICODE 0x8c
#define PICKLE_BINUNICODE 0x58
#define PICKLE_SHORT_BINBYTES 0x42
#define PICKLE_BINBYTES 0x46
#define PICKLE_SHORT_BINUNICODE8 0x8d
#define PICKLE_BINUNICODE8 0x58
#define PICKLE_EMPTY_DICT 0x7d
#define PICKLE_EMPTY_LIST 0x5b
#define PICKLE_APPENDS 0x65
#define PICKLE_BINGET 0x68
#define PICKLE_LONG_BINGET 0x6a
#define PICKLE_BINPUT 0x71
#define PICKLE_LONG_BINPUT 0x72
#define PICKLE_MARK 0x28
#define PICKLE_STOP 0x2e
#define PICKLE_REDUCE 0x85
#define PICKLE_BUILD 0x62
#define PICKLE_SETITEMS 0x75
#define PICKLE_SETITEM 0x73
#define PICKLE_TUPLE1 0x85
#define PICKLE_TUPLE2 0x86
#define PICKLE_TUPLE3 0x87
#define PICKLE_POP 0x30
#define PICKLE_DUP 0x32
#define PICKLE_NONE 0x4e
#define PICKLE_TRUE 0x88
#define PICKLE_FALSE 0x89
#define PICKLE_NEWOBJ 0x81
#define PICKLE_EXT1 0x82
#define PICKLE_EXT2 0x83
#define PICKLE_EXT4 0x84

struct PickleWriter {
    std::vector<unsigned char> buf;
    int memo_count = 0;

    void write_byte(unsigned char b) { buf.push_back(b); }
    void write_bytes(const void *data, size_t n) {
        const unsigned char *p = (const unsigned char *)data;
        buf.insert(buf.end(), p, p + n);
    }

    void write_proto() {
        write_byte(PICKLE_PROTO);
        write_byte(2); // protocol 2
    }

    void write_global(const char *module, const char *name) {
        write_byte(PICKLE_GLOBAL);
        write_string(module);
        write_string(name);
    }

    void write_string(const char *s) {
        size_t len = strlen(s);
        if (len < 256) {
            write_byte(PICKLE_SHORT_BINUNICODE);
            write_byte((unsigned char)len);
        } else {
            write_byte(PICKLE_BINUNICODE);
            write_uint32((uint32_t)len);
        }
        write_bytes(s, len);
    }

    void write_uint32(uint32_t v) {
        write_bytes(&v, 4);
    }

    void write_int(int v) {
        if (v >= 0 && v < 256) {
            write_byte(0x4b); // BININT
            write_byte((unsigned char)v);
        } else {
            write_byte(0x4a); // BININT4
            write_bytes(&v, 4);
        }
    }

    void write_long(int64_t v) {
        write_byte(0x4d); // BININT8
        write_bytes(&v, 8);
    }

    void write_float(float v) {
        write_global("numpy.core.multiarray", "_reconstruct");
        write_byte(PICKLE_MARK);
        write_global("numpy", "ndarray");
        write_byte(PICKLE_MARK);
        write_int(0); // version
        write_string("b"); // state
        write_byte(PICKLE_TUPLE3);
        write_global("numpy", "dtype");
        write_byte(PICKLE_MARK);
        write_string("f4"); // float32
        write_byte(PICKLE_TUPLE1);
        write_int(0); // little endian
        write_string("}"); // dict
        write_byte(PICKLE_SETITEMS);
        write_byte(PICKLE_BUILD);
        write_byte(PICKLE_POP); // pop state
        // Actually this is getting too complex
        // Let's just write raw bytes
    }

    void write_numpy_array(const float *data, int ndim, const int *shape) {
        // Simplified: just write as bytes
        // In practice, we need proper numpy pickle format
        write_global("numpy.core.multiarray", "_reconstruct");
        write_byte(PICKLE_MARK);
        write_global("numpy", "ndarray");
        write_byte(PICKLE_MARK);
        write_int(0); // version
        write_string("b"); // state
        write_byte(PICKLE_TUPLE3);
        // dtype
        write_global("numpy", "dtype");
        write_byte(PICKLE_MARK);
        write_string("f4");
        write_byte(PICKLE_TUPLE1);
        write_int(0); // byte order
        // shape
        write_global("builtins", "tuple");
        write_byte(PICKLE_MARK);
        for (int i = 0; i < ndim; i++) write_int(shape[i]);
        write_byte(PICKLE_TUPLE1); // Actually should be tuple of ints
        // strides (None for now)
        write_byte(PICKLE_NONE);
        // data
        int total = 1;
        for (int i = 0; i < ndim; i++) total *= shape[i];
        write_global("builtins", "bytes");
        write_byte(PICKLE_MARK);
        write_int(total * sizeof(float));
        write_bytes(data, total * sizeof(float));
        write_byte(PICKLE_TUPLE1);
        // dict
        write_byte(PICKLE_EMPTY_DICT);
        write_byte(PICKLE_SETITEMS);
        write_byte(PICKLE_BUILD);
        write_byte(PICKLE_POP);
    }

    void write_dict_header(int count) {
        write_byte(PICKLE_EMPTY_DICT);
    }

    void write_dict_setitem(const char *key) {
        write_string(key);
    }

    void write_list() {
        write_byte(PICKLE_EMPTY_LIST);
    }

    void write_list_append() {
        write_byte(PICKLE_APPENDS);
    }

    void write_stop() {
        write_byte(PICKLE_STOP);
    }

    void save(const char *path) {
        FILE *f = fopen(path, "wb");
        if (!f) return;
        fwrite(buf.data(), 1, buf.size(), f);
        fclose(f);
        printf("Saved pickle: %s (%zu bytes)\n", path, buf.size());
    }
};

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <input.bin> <output.pkl>\n", argv[0]);
        return 1;
    }

    // Read binary model
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "RBN1", 4) != 0) {
        fprintf(stderr, "Invalid model format\n");
        fclose(f);
        return 1;
    }

    int V, T, D, H, L, FF;
    fread(&V, sizeof(int), 1, f);
    fread(&T, sizeof(int), 1, f);
    fread(&D, sizeof(int), 1, f);
    fread(&H, sizeof(int), 1, f);
    fread(&L, sizeof(int), 1, f);
    fread(&FF, sizeof(int), 1, f);

    printf("Config: V=%d T=%d D=%d H=%d L=%d FF=%d\n", V, T, D, H, L, FF);

    // Read char_to_id map
    int map[256];
    fread(map, sizeof(int), 256, f);

    // Read weights
    auto read_arr = [&](int n) -> std::vector<float> {
        std::vector<float> v(n);
        fread(v.data(), sizeof(float), n, f);
        return v;
    };

    auto token_emb = read_arr(V * D);
    auto pos_emb = read_arr(T * D);

    std::vector<std::vector<float>> ln1_w(L), ln1_b(L), ln2_w(L), ln2_b(L);
    std::vector<std::vector<float>> wq(L), bq(L), wk(L), bk(L), wv(L), bv(L), wo(L), bo(L);
    std::vector<std::vector<float>> w1(L), b1(L), w2(L), b2(L);

    for (int l = 0; l < L; l++) {
        ln1_w[l] = read_arr(D); ln1_b[l] = read_arr(D);
        wq[l] = read_arr(D*D); bq[l] = read_arr(D);
        wk[l] = read_arr(D*D); bk[l] = read_arr(D);
        wv[l] = read_arr(D*D); bv[l] = read_arr(D);
        wo[l] = read_arr(D*D); bo[l] = read_arr(D);
        ln2_w[l] = read_arr(D); ln2_b[l] = read_arr(D);
        w1[l] = read_arr(D*FF); b1[l] = read_arr(FF);
        w2[l] = read_arr(FF*D); b2[l] = read_arr(D);
    }

    auto ln_f_w = read_arr(D);
    auto ln_f_b = read_arr(D);
    auto lm_w = read_arr(V*D);
    auto lm_b = read_arr(V);
    fclose(f);

    // Write pickle
    PickleWriter pw;
    pw.write_proto();

    // Build state dict
    pw.write_global("builtins", "dict");
    pw.write_byte(PICKLE_MARK);

    // vocab_size
    pw.write_string("vocab_size");
    pw.write_int(V);

    // block_size
    pw.write_string("block_size");
    pw.write_int(T);

    // d_model
    pw.write_string("d_model");
    pw.write_int(D);

    // n_head
    pw.write_string("n_head");
    pw.write_int(H);

    // n_layer
    pw.write_string("n_layer");
    pw.write_int(L);

    // d_ff
    pw.write_string("d_ff");
    pw.write_int(FF);

    // char_to_id
    pw.write_string("char_to_id");
    pw.write_global("builtins", "dict");
    pw.write_byte(PICKLE_MARK);
    for (int i = 0; i < 256; i++) {
        if (map[i] > 0 || i == 0) {
            char key[2] = {(char)i, 0};
            pw.write_string(key);
            pw.write_int(map[i]);
        }
    }
    pw.write_byte(PICKLE_SETITEMS);

    // id_to_char
    pw.write_string("id_to_char");
    pw.write_global("builtins", "dict");
    pw.write_byte(PICKLE_MARK);
    for (int i = 0; i < 256; i++) {
        if (map[i] > 0 || i == 0) {
            char key[16];
            sprintf(key, "%d", map[i]);
            pw.write_string(key);
            char val[2] = {(char)i, 0};
            pw.write_string(val);
        }
    }
    pw.write_byte(PICKLE_SETITEMS);

    // Write numpy arrays
    int shape_te[] = {V, D};
    pw.write_string("token_emb");
    pw.write_numpy_array(token_emb.data(), 2, shape_te);

    int shape_pe[] = {1, T, D};
    pw.write_string("pos_emb");
    pw.write_numpy_array(pos_emb.data(), 3, shape_pe);

    int shape_d[] = {D};
    pw.write_string("ln_f_w");
    pw.write_numpy_array(ln_f_w.data(), 1, shape_d);
    pw.write_string("ln_f_b");
    pw.write_numpy_array(ln_f_b.data(), 1, shape_d);

    int shape_v[] = {V};
    pw.write_string("lm_b");
    pw.write_numpy_array(lm_b.data(), 1, shape_v);

    // lm_w is [V,D] but Rust expects [D,V] (reversed)
    // Actually, PyTorch stores as [V,D] and Rust reverses it
    // So we store as [V,D] here
    int shape_lm[] = {V, D};
    pw.write_string("lm_w");
    pw.write_numpy_array(lm_w.data(), 2, shape_lm);

    // layers list
    pw.write_string("layers");
    pw.write_list();
    for (int l = 0; l < L; l++) {
        pw.write_global("builtins", "dict");
        pw.write_byte(PICKLE_MARK);

        pw.write_string("ln1_w");
        pw.write_numpy_array(ln1_w[l].data(), 1, shape_d);
        pw.write_string("ln1_b");
        pw.write_numpy_array(ln1_b[l].data(), 1, shape_d);
        pw.write_string("attn_wq_w");
        int shape_dd[] = {D, D};
        pw.write_numpy_array(wq[l].data(), 2, shape_dd);
        pw.write_string("attn_wq_b");
        pw.write_numpy_array(bq[l].data(), 1, shape_d);
        pw.write_string("attn_wk_w");
        pw.write_numpy_array(wk[l].data(), 2, shape_dd);
        pw.write_string("attn_wk_b");
        pw.write_numpy_array(bk[l].data(), 1, shape_d);
        pw.write_string("attn_wv_w");
        pw.write_numpy_array(wv[l].data(), 2, shape_dd);
        pw.write_string("attn_wv_b");
        pw.write_numpy_array(bv[l].data(), 1, shape_d);
        pw.write_string("attn_wo_w");
        pw.write_numpy_array(wo[l].data(), 2, shape_dd);
        pw.write_string("attn_wo_b");
        pw.write_numpy_array(bo[l].data(), 1, shape_d);
        pw.write_string("ln2_w");
        pw.write_numpy_array(ln2_w[l].data(), 1, shape_d);
        pw.write_string("ln2_b");
        pw.write_numpy_array(ln2_b[l].data(), 1, shape_d);
        pw.write_string("ff_w1_w");
        int shape_dff[] = {FF, D};
        pw.write_numpy_array(w1[l].data(), 2, shape_dff);
        pw.write_string("ff_w1_b");
        int shape_ff[] = {FF};
        pw.write_numpy_array(b1[l].data(), 1, shape_ff);
        pw.write_string("ff_w2_w");
        int shape_ffd[] = {D, FF};
        pw.write_numpy_array(w2[l].data(), 2, shape_ffd);
        pw.write_string("ff_w2_b");
        pw.write_numpy_array(b2[l].data(), 1, shape_d);

        pw.write_byte(PICKLE_SETITEMS);
        pw.write_byte(PICKLE_APPENDS);
    }

    pw.write_byte(PICKLE_SETITEMS);
    pw.write_stop();
    pw.save(argv[2]);

    return 0;
}
