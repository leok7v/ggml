#define main run
#define gpt_neox_model_load ggml_gpt_neox_model_load
#include "main.cpp"
#include "rt.h"
#undef main
#undef gpt_neox_model_load

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
        println("n_vocab = %d", hparams.n_vocab);
        println("n_ctx   = %d", hparams.n_ctx);
        println("n_embd  = %d", hparams.n_embd);
        println("n_head  = %d", hparams.n_head);
        println("n_layer = %d", hparams.n_layer);
        println("n_rot   = %d", hparams.n_rot);
        println("par_res = %d", hparams.par_res);
        println("ftype   = %d", hparams.ftype);
        println("qntvr   = %d", qntvr);
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
        println("");
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
        println("done");
        println("model size = %8.2f MB / num tensors = %d", total_size/1024.0/1024.0, n_tensors);
    }
    fin.close();
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();
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
    println("seed = %d", params.seed);
    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }
    int64_t t_load_us = 0;
    gpt_vocab vocab;
    gpt_neox_model model;
    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        if (!gpt_neox_model_load(params.model, model, vocab)) {
            println("failed to load model from '%s'", params.model.c_str());
            return 1;
        }
        t_load_us = ggml_time_us() - t_start_us;
        test_gpt_tokenizer(vocab, params.token_test);
    }
    int n_past = 0;
    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;
    std::vector<float> logits;
    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);
    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());
    println("number of tokens in prompt = %zu", embd_inp.size());
    for (int i = 0; i < embd_inp.size(); i++) {
        println("token[%d] = %6d, %s", i, embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    println("");
    std::vector<gpt_vocab::id> embd;
    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gpt_neox_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);
    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();
            if (!gpt_neox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                println("Failed to predict");
                return 1;
            }
            t_predict_us += ggml_time_us() - t_start_us;
        }
        n_past += embd.size();
        embd.clear();
        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const int n_vocab = model.hparams.n_vocab;
            gpt_vocab::id id = 0;
            {
                const int64_t t_start_sample_us = ggml_time_us();
                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
                t_sample_us += ggml_time_us() - t_start_sample_us;
            }
            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }
        // display text
        for (auto id : embd) {
            println("%s", vocab.id_to_token[id].c_str());
        }
        // end of text token
        if (embd.back() == 0) {
            break;
        }
    }
    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        println("");
        println("mem per token = %8zu bytes", mem_per_token);
        println("    load time = %8.2f ms", t_load_us/1000.0f);
        println("  sample time = %8.2f ms", t_sample_us/1000.0f);
        println(" predict time = %8.2f ms / %.2f ms per token", t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        println("   total time = %8.2f ms", (t_main_end_us - t_main_start_us)/1000.0f);
    }
    ggml_free(model.ctx);
    return 0;
}
