#define main run
#define gpt_neox_model_load ggml_gpt_neox_model_load
#include "main.cpp"
#include "rt.h"
#undef main
#undef gpt_neox_model_load
#ifdef WIN32
#include <Windows.h>
#include <WinCon.h>
#endif
#include <set>

#ifdef WIN32
// cyan on blue background:
// FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_BLUE
// White background Black text:
// BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED
static void set_text_color(uint32_t foreground, uint32_t background) {
    uint16_t a = 0;
    int r = (foreground >>  0) & 0x3;
    int g = (foreground >>  8) & 0x3;
    int b = (foreground >> 16) & 0x3;
    if (r > 1 || g > 1 || b > 1) { a |= FOREGROUND_INTENSITY; }
    if (r > 0) { a |= FOREGROUND_RED; }
    if (g > 0) { a |= FOREGROUND_GREEN; }
    if (b > 0) { a |= FOREGROUND_BLUE; }
    r = (background >>  0) & 0x3;
    g = (background >>  8) & 0x3;
    b = (background >> 16) & 0x3;
    if (r > 1 || g > 1 || b > 1) { a |= BACKGROUND_INTENSITY; }
    if (r > 0) { a |= BACKGROUND_RED; }
    if (g > 0) { a |= BACKGROUND_GREEN; }
    if (b > 0) { a |= BACKGROUND_BLUE; }
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, a);
}

#else
static void set_text_color(uint32_t foreground, uint32_t background) {
    (void)foreground; (void)background;
    // TODO: vt100 ESC colors? Curses?
}
#endif

// load the model's weights from a file
bool gpt_neox_model_load(const std::string & fname, gpt_neox_model & model, gpt_vocab & vocab) {
    println("loading model from '%s' - please wait ...", fname.c_str());
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        println("failed to open '%s'", fname.c_str());
        return false;
    }
    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            println("invalid model file '%s' (bad magic)", fname.c_str());
            return false;
        }
    }
    // load hparams
    {
        auto & hparams = model.hparams;
        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.par_res, sizeof(hparams.par_res));
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));
        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        traceln("n_vocab = %d", hparams.n_vocab);
        traceln("n_ctx   = %d", hparams.n_ctx);
        traceln("n_embd  = %d", hparams.n_embd);
        traceln("n_head  = %d", hparams.n_head);
        traceln("n_layer = %d", hparams.n_layer);
        traceln("n_rot   = %d", hparams.n_rot);
        traceln("par_res = %d", hparams.par_res);
        traceln("ftype   = %d", hparams.ftype);
        traceln("qntvr   = %d", qntvr);
        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }
    // load vocab
    {
        const int32_t n_vocab = model.hparams.n_vocab;
        std::string word;
        std::vector<char> buf(128);
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));
            buf.resize(len);
            fin.read((char *) buf.data(), len);
            word.assign(buf.data(), len);
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }
    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        println("invalid model file '%s' (bad ftype value %d)",
                fname.c_str(), model.hparams.ftype);
        return false;
    }
    auto & ctx = model.ctx;
    size_t ctx_size = 0;
    {
        const auto & hparams = model.hparams;
        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // wte

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype);           // lmh_g
        //ctx_size +=        n_vocab*ggml_type_sizef(GGML_TYPE_F32); // lmh_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(wtype));         // c_attn_attn_w
        ctx_size += n_layer*(       3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_attn_attn_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype));         // c_attn_proj_w
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_attn_proj_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_fc_w
        ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_proj_w
        ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 16*n_layer)*512; // object overhead

        println("ggml ctx size = %6.2f MB", ctx_size/(1024.0*1024.0));
    }
    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        model.ctx = ggml_init(params);
        if (!model.ctx) {
            println("ggml_init() failed");
            return false;
        }
    }
    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;
        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_vocab = hparams.n_vocab;
        model.layers.resize(n_layer);
        model.wte    = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.lmh_g  = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        //model.lmh_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);
        // map by name
        model.tensors["gpt_neox.embed_in.weight"] = model.wte;
        model.tensors["gpt_neox.final_layer_norm.weight"] = model.ln_f_g;
        model.tensors["gpt_neox.final_layer_norm.bias"]   = model.ln_f_b;
        model.tensors["embed_out.weight"] = model.lmh_g;
        //model.tensors["lm_head.bias"]   = model.lmh_b;
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];
            layer.ln_1_g          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_1_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.c_attn_attn_w   = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);
            layer.c_attn_proj_w   = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_proj_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_g          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.c_mlp_fc_w      = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);
            layer.c_mlp_proj_w    = ggml_new_tensor_2d(ctx, wtype,         4*n_embd,   n_embd);
            layer.c_mlp_proj_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            // map by name
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight"] = layer.ln_1_g;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias"]   = layer.ln_1_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.weight"] = layer.c_attn_attn_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.bias"]   = layer.c_attn_attn_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight"] = layer.c_attn_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias"]   = layer.c_attn_proj_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight"] = layer.ln_2_g;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias"]   = layer.ln_2_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"] = layer.c_mlp_fc_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]   = layer.c_mlp_fc_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"] = layer.c_mlp_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]   = layer.c_mlp_proj_b;
        }
    }
    // key + value memory
    {
        const auto & hparams = model.hparams;
        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int64_t n_mem      = n_layer*n_ctx;
        const int64_t n_elements = n_embd*n_mem;
        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);
        println("memory_size = %8.2f MB, n_mem = %" PRId64 "", memory_size/1024.0/1024.0, n_mem);
    }
    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;
        traceln("");
        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;
            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
            if (fin.eof()) {
                break;
            }
            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }
            std::string name(length, 0);
            fin.read(&name[0], length);
            if (model.tensors.find(name.data()) == model.tensors.end()) {
                println("unknown tensor '%s' in model file", name.data());
                return false;
            }
            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                println("tensor '%s' has wrong size in model file", name.data());
                return false;
            }
            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                println("tensor '%s' has wrong shape in model file: got [%5d, %5d], expected [%5d, %5d]",
                        name.data(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }
            // for debugging
            if (0) {
//              if ((ggml_nbytes(tensor)/1024.0/1024.0) > 1) {
                    println("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes", name.data(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
//              }
            }
            const size_t bpe = ggml_type_size(ggml_type(ttype));
            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                println("tensor '%s' has wrong size in model file: got %zu, expected %zu",
                        name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }
            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
//              println(".");
//              fflush(stdout);
            }
        }
        traceln("done");
        println("model size = %8.2f MB / num tensors = %d", total_size/1024.0/1024.0, n_tensors);
    }
    fin.close();
    return true;
}

#if 0
#define system_prompt "<|SYSTEM|># StableLM Tuned (Alpha version) "     \
"- StableLM is a helpful and harmless open-source AI language model "   \
  "developed by StabilityAI. "                                          \
"- StableLM is excited to be able to help the user, but will refuse "   \
  "to do anything that could be considered harmful to the user. "       \
"- StableLM is more than just an information source, StableLM is also " \
  "able to write poetry, short stories, and make jokes. "               \
"- StableLM will refuse to participate in anything that could harm a "  \
  "human. "
#else
#define system_prompt ""
#endif

// #define starting_prompt system_prompt                                   \
// "key: markets\n"                                                        \
// "tweet: Take feedback from nature and markets, not from people.\n"      \
// "###\n"                                                                 \
// "key: children\n"                                                       \
// "tweet: Maybe we die so we can come back as children.\n"                \
// "###\n"                                                                 \
// "key: startups\n"                                                       \
// "tweet: Startups shouldn’t worry about how to put out fires, "          \
// "they should worry about how to start them.\n"                          \
// "###\n"                                                                 \
// "key: hugging face\n"                                                   \
// "tweet: "

#define starting_prompt system_prompt \
    "key: question\n"                 \
    "tweet: Answer.\n"                \
    "###\n"

#include <Windows.h>

static std::vector<gpt_vocab::id> user_input(gpt_vocab &vocab) {
    char text[4096] = {0};
    std::vector<gpt_vocab::id> append;
    while (text[0] == 0 && append.size() == 0) {
        set_text_color(0x000002, 0x000000); // bright red on black:
        printf("\nUser: "); fflush(stdout);
        set_text_color(0x010101, 0x000000); // white on black:
        fgets(text, countof(text) , stdin);
        if (strstr(text, "qiut") == text || strstr(text, "exit")) {
            return std::vector<gpt_vocab::id>();
        } else {
            return ::gpt_tokenize(vocab, text);
        }
    }
    return std::vector<gpt_vocab::id>();
}

static void print_text(gpt_vocab &vocab, std::vector<gpt_vocab::id> &embd, int pos) {
    set_text_color(0x000200, 0x000000); // bright green on black:
    int k = 0;
    for (auto id : embd) {
        const char* token = vocab.id_to_token[id].c_str();
        if (k >= pos) {
            printf("%s", token);
        }
        k++;
    }
    fflush(stdout);
    set_text_color(0x010101, 0x000000); // white on black:
}

enum {
    id_end_of_text =     0, // "<|endoftext|>"
    id_padding     =     1, // "<|padding|>"
    id_long_hash   = 22902, // "################################"
    id_less        = 16375, // "<"
    id_bar          =   93, // "|" less+bar "<|"
    id_open        = 41533, // "|<"
    id_close       = 49651, // "|>"
    id_human       = 13961, // "human"
    id_human1      = 22705, // "Human"
    id_redit       = 12289, // "redit"
    id_twitter     = 16705, // "twitter"
    id_twitter1    = 31068, // "Twitter"
    id_column      =    27, // "user"
    id_user        =  4537, // "user"
    id_user1       = 23131, // "USER"
    id_user2       =  6989, // "User"
    id_system      = 10394, // "system"
    id_system1     =  7761, // "System"
    id_system2     = 47146, // "SYSTEM"
    id_wikipedia   = 25842, // "wikipedia"
    id_last        = 50276, // "\x20\x20" aka "  " of 50688
    id_https       =  3614, // 'https'
    id_http       =  2413, // 'http'
    id_url         =  1358, // '://'
    // there is a lot of multi spaces intersperced with LF
//  id_space       =   209, // "\x20" aka " "
//  id_lf          =   187, // "\n"
};

// 2239 ' >'

static void dictionary(gpt_vocab &vocab) {
/*
    for (int id = vocab.id_to_token.size() - 1; id > 0; id--) {
        if (vocab.id_to_token[id].c_str()[0] != 0) {
            traceln("last: %d of %d \"%s\"", id,
                (int)vocab.id_to_token.size(), vocab.id_to_token[id].c_str());
            break;
        }
    }
*/
#if 0
    (void)MB_ERR_INVALID_CHARS;
    for (auto e : vocab.token_to_id) {
        auto id = e.second;
        const char* token = e.first.c_str();
        std::string s = std::string(e.first) + "";
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c){ return std::tolower(c); });
        const char* u = s.c_str(); // uncased
        if (strcmp(u, "\x20") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "user") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "system") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "tweet") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "twitter") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "redit") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "wikipedia") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "assistant") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "human") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "tweet") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
//      if (strcmp(u, ":") == 0) {
//          traceln("id: %d token: \"%s\"", id, token);
//      }
//      if (strstr(u, "<|") != null || strstr(u, ">|") != null || strstr(u, "#") != null) {
//          traceln("id: %d token: \"%s\"", id, token);
//      }
//      if (strcmp(u, ":") == 0) {
//          traceln("id: %d token: \"%s\"", id, token);
//      }
        if (strstr(u, ">") != null || strstr(u, "<") != null) {
            traceln("id: %d token: \"%s\"", id, token);
        }
        if (strcmp(u, "|") == 0) {
            traceln("id: %d token: \"%s\"", id, token);
        }
    }
#else
    (void)vocab;
#endif
}

gpt_vocab::id sample_top_k_top_p(
        const gpt_vocab & vocab,
        const std::set<gpt_vocab::id> &except,
        const std::set<gpt_vocab::id> &stop,
        const float * logits,
        int    top_k,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();
    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);
    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; i++) {
            if (except.count(i) == 0) {
                logits_id.push_back(std::make_pair(logits[i]*scale, i));
            } else {
//              traceln("skipped: %d \"%s\"", i, vocab.id_to_token.at(i).c_str());
            }
        }
    }
    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });
    logits_id.resize(top_k);
    bool end_of_text = false;
    for (int i = 0; i < logits_id.size() && !end_of_text; i++) {
        int id = logits_id[i].second;
        end_of_text = stop.count(id) > 0;
//      if (end_of_text) {
//          traceln("id: %d token: \"%s\"", id, vocab.id_to_token.at(id).c_str());
//          debugbreak();
//      }
    }
    if (!end_of_text) {
        double maxl = -INFINITY;
        for (const auto & kv : logits_id) {
            maxl = max(maxl, kv.first);
        }
        // compute probs for the top K tokens
        std::vector<double> probs;
        probs.reserve(logits_id.size());
        double sum = 0.0;
        for (const auto & kv : logits_id) {
            double p = exp(kv.first - maxl);
            probs.push_back(p);
            sum += p;
        }
        // normalize the probs
        for (auto & p : probs) { p /= sum; }
        if (top_p < 1.0f) {
            double cumsum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                cumsum += probs[i];
                if (cumsum >= top_p) {
                    top_k = i + 1;
                    probs.resize(top_k);
                    logits_id.resize(top_k);
                    break;
                }
            }
            cumsum = 1.0/cumsum;
            for (int i = 0; i < (int) probs.size(); i++) {
                probs[i] *= cumsum;
            }
        }
//      traceln("");
//      for (int i = 0; i < (int) probs.size(); i++) {
//          auto id = logits_id[i].second;
//          traceln("[%d]: %d '%s' %f\n", i, id, vocab.id_to_token.at(id).c_str(), probs[i]);
//      }
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist(rng);
        return logits_id[idx].second;
    } else {
//      traceln("");
//      for (int i = 0; i < (int) logits_id.size(); i++) {
//          auto id = logits_id[i].second;
//          traceln("[%d]: %d '%s' %f\n", i, id, vocab.id_to_token.at(id).c_str(), logits_id[i].first);
//      }
        return id_end_of_text;
    }
}

// -p "Question: The best way to learn a new foreign language is? Short Answer: "

int main(int argc, char ** argv) {
//  run(argc, argv);
    ggml_time_init();
//  enum { CP_UTF8 = 65001 };
    SetConsoleOutputCP(CP_UTF8);
    std::string prompt = "";
    const int64_t t_main_start_us = ggml_time_us();
    gpt_params params;
//  params.model = "models/stablelm-base-alpha-3b/ggml-model-f16.bin";
    params.model = "models/ggml-model-stablelm-base-alpha-3b-q4_0.bin";
    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }
    if (params.seed < 0) {
        params.seed = time(NULL);
    }
    traceln("seed = %d", params.seed);
    std::mt19937 rng(params.seed);
    int64_t t_load_us = 0;
    gpt_vocab vocab;
    gpt_neox_model model;
    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        if (!gpt_neox_model_load(params.model, model, vocab)) {
            traceln("failed to load model from '%s'", params.model.c_str());
            return 1;
        }
        t_load_us = ggml_time_us() - t_start_us;
//      test_gpt_tokenizer(vocab, params.token_test);
    }
    // RoPE stands for "Relative Positional Encoding."
    // n_past: https://arxiv.org/pdf/2104.09864.pdf
    int n_past = 0;
    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;
    std::vector<float> logits;
    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    int64_t t_start_us = ggml_time_us();
    fatal_if(!gpt_neox_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token));
    t_predict_us += ggml_time_us() - t_start_us;
    dictionary(vocab);
    std::vector<gpt_vocab::id> embd;
    std::set<gpt_vocab::id> except;
    std::set<gpt_vocab::id> stop;
    const std::set<gpt_vocab::id> empty;
    if (!params.prompt.empty()) {
        prompt = params.prompt + "\n";
        embd = ::gpt_tokenize(vocab, prompt);
    }
    // neox::StableLM has a lot of User: User1: User2: User3: junk
    except.insert(id_user);
    except.insert(id_user1);
    except.insert(id_user2);
    // don't want any halucinations about urlss
    except.insert(id_https);
    except.insert(id_http);
    except.insert(id_url);
    stop.insert(id_end_of_text);
    params.temp = 0.9f;
    int stopid = 0; // vocab.token_to_id["###"];
    n_past = 0;
    bool done = false;
    bool ask = prompt.empty();
    int last_id = -1;
    int answer_word_count = 0;
    int64_t predictions = 0;
    int64_t samples = 0;
    while (!done) {
        if (!ask) {
            if (answer_word_count > 0) {
                print_text(vocab, embd, 0);
            }
            t_start_us = ggml_time_us();
            fatal_if(!gpt_neox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token));
            t_predict_us += ggml_time_us() - t_start_us;
            n_past += embd.size();
            predictions += embd.size();
            embd.clear();
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const int n_vocab = model.hparams.n_vocab;
            gpt_vocab::id id = 0;
            const int64_t t_start_sample_us = ggml_time_us();
            id = sample_top_k_top_p(vocab, except, answer_word_count > 100 ? stop : empty,
                logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
            t_sample_us += ggml_time_us() - t_start_sample_us;
            samples++;
            // add it to the context
            if (id == id_end_of_text) {
                ask = true;
            } else if (id > id_last) {
                traceln("id: %d > id_last", id);
                ask = true;
            } else if (answer_word_count > 300) {
                ask = true; // too long
            } else if (answer_word_count > 150 && vocab.id_to_token[id].c_str()[0] == '\n') {
                ask = true; // enough
                print_text(vocab, embd, 0);
            } else if (answer_word_count > 150 && vocab.id_to_token[id].c_str()[0] == '.') {
                ask = true; // enough
                print_text(vocab, embd, 0);
            } else if (id == last_id) {
                ask = true; // stutering
            } else {
                embd.push_back(id);
                last_id = id;
                answer_word_count++;
            }
//          traceln("---- added id: %d", id);
        } else if (params.prompt.empty()) {
//          traceln("answer_word_count: %d", answer_word_count);
            std::vector<gpt_vocab::id> u = user_input(vocab);
            if (u.size() == 0) { break; }
            embd.insert(embd.end(), u.begin(), u.end());
            n_past = 0;
            ask = false;
            answer_word_count = 0;
            last_id = -1;
        } else if (!params.prompt.empty()) {
            done = true;
        }
    }
    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        traceln("");
        traceln("mem per token = %8zu bytes", mem_per_token);
        traceln("    load time = %8.2f ms", t_load_us/1000.0f);
        traceln("  sample time = %8.2f ms", t_sample_us/1000.0f/samples);
        traceln(" predict time = %8.2f ms / %.2f ms per token", t_predict_us/1000.0f, t_predict_us/1000.0f/predictions);
        traceln("   total time = %8.2f ms", (t_main_end_us - t_main_start_us)/1000.0f);
    }
    ggml_free(model.ctx);
    return 0;
}

/*

50278, 50279, 50277, 1, 0
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

*/