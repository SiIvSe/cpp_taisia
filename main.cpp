#include "llama.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "nlohmann/json.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

using json = nlohmann::json;
using namespace std;

// Простая функция токенизации без common.h
vector<llama_token> tokenize(llama_context* ctx, const string& text, bool add_bos) {
    const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(ctx));
    vector<llama_token> tokens(256);
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
        tokens.data(), tokens.size(), add_bos, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
            tokens.data(), tokens.size(), add_bos, true);
    }
    tokens.resize(n_tokens);
    return tokens;
}

// Простая функция конвертации токена в строку
string token_to_piece(llama_context* ctx, llama_token token) {
    const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(ctx));
    vector<char> buf(32);
    int32_t len = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    if (len < 0) {
        buf.resize(-len);
        len = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
    }
    return string(buf.data(), len);
}

void app_histary(string filename, string prompt, string respone) {
    json resp = {
        {"Запрос", prompt},
        {"Ответ", respone}
    };
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << resp.dump() << "/n";
        file.close();
    } 
    else {
        cerr << "Не удалось открыть файл" << endl;
    }
}

int main() {
    // Устанавливаем UTF-8 кодировку для консоли Windows
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // Вывод в UTF-8
    SetConsoleCP(CP_UTF8);        // Ввод в UTF-8 (ИСПРАВЛЕНИЕ!)
    #endif
    // Отключаем debug-логи CUDA
    llama_log_set([](ggml_log_level level, const char* text, void* user_data) {
        if (level >= GGML_LOG_LEVEL_INFO) {  // Показываем только важные сообщения
            fputs(text, stderr);
        }
        }, nullptr);

    string model_path = "C:/Users/SiIvSe/.lmstudio/models/bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf";
    string prompt = "Привет! Напиши код на C++, который будет записывать переменую в текстовый документ.";
    int n_predict = 1024;

    llama_backend_init();

    // Загрузка модели
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;

    llama_model* model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        cerr << "Ошибка загрузки модели: " << model_path << endl;
        return 1;
    }

    // Параметры контекста
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 32768;
    cparams.n_batch = 1024;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        cerr << "Ошибка создания контекста" << endl;
        llama_free_model(model);
        return 1;
    }
    for (int j = 0; j < 100; j++) {
        // Токенизация
        cout << "__________________________________________________________________________________________________________________________" << endl;
        cout << "Ввод: ";
        string prompt;
        getline(cin, prompt);

        if (prompt == "выход" || prompt == ".") break;
        if (prompt.empty()) continue;

        //ФОРМАТ С СИСТЕМНЫМ ПРОМПТОМ
        string system_prompt = "Ты — Таисия, дружелюбный помощник. Отвечай на вопросы пользователя по существу, не выдумывай лишнего.";
        string formatted_prompt = "[INST] " + system_prompt + "\n\nПользователь: " + prompt + " [/INST]";

        vector<llama_token> tokens = tokenize(ctx, formatted_prompt, true);

        // Обработка промпта
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

        if (llama_decode(ctx, batch) != 0) {
            cerr << "Ошибка при кодировании промпта" << endl;
            llama_free(ctx);
            llama_free_model(model);
            return 1;
        }

        // Создаем сэмплер
        llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        cout << "Запрос: " << prompt << "\n\nОтвет модели:\n";

        // Генерация ответа
        vector<string> answer = {"Модель: "};
        for (int i = 0; i < n_predict; i++) {
            llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

            const llama_vocab* vocab = llama_model_get_vocab(model);
            if (new_token == llama_token_eos(vocab)) {
                break;
            }

            string piece = token_to_piece(ctx, new_token);
            cout << piece << flush;
            answer.push_back(piece);

            batch = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch) != 0) {
                cerr << "\nОшибка при генерации" << endl;
                break;
            }
        }

        string respone;
        for (const auto& s : answer) respone += s;
        app_histary("C:/LLM_CPP/history.jsonl", prompt, respone);
        cout << "\n\n[Завершено]" << endl;


        llama_sampler_free(sampler);
        cout << "__________________________________________________________________________________________________________________________" << endl;
    }
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}