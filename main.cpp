#include "llama.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
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
        file << resp.dump() << "\n";
        file.close();
    }
    else {
        cerr << "Не удалось открыть файл" << endl;
    }
}

void clear_file(string filename) {
    ofstream file(filename, ios::trunc);

    if (file.is_open()) {
        file.close(); // Файл успешно очищен и закрыт
        cout << "Файл очищен!" << endl;
    }
    else {
        cerr << "Не удалось открыть файл." << endl;
    }
}

void app_vospominania(string filename, string vosp) {
    clear_file("C:/LLM_CPP/history.jsonl");
    ofstream file(filename, ios::app);
    if (file.is_open()) {
        file << vosp << "\n";
        file.close();
    }
    else {
        cerr << "Не удалось открыть файл" << endl;
    }
}

string get_history(string filename) {
    ifstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл" << endl;
        return "";
    }
    ostringstream history;
    history << file.rdbuf(); 
    file.close();
    return history.str();
}


string massag(string prompt, string history, int n_predict, llama_model* model, llama_context* ctx, string sys_p) {

    //ФОРМАТ С СИСТЕМНЫМ ПРОМПТОМ
    string system_prompt = sys_p;
    string formatted_prompt = "[INST] " + system_prompt + "\nВоспоминания: " + history + "\n\nПользователь: " + prompt + " [/INST]";

    vector<llama_token> tokens = tokenize(ctx, formatted_prompt, true);

    // Обработка промпта
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    if (llama_decode(ctx, batch) != 0) {
        cerr << "Ошибка при кодировании промпта" << endl;
        llama_free(ctx);
        llama_free_model(model);
        //return 1;
    }

    // Создаем сэмплер
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    cout << "Запрос: " << prompt << "\n\nОтвет модели:\n";

    // Генерация ответа
    vector<string> answer = { "Ассиситент: " };
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
    string histry = "Пользователь: " + prompt + "Таисия: " + respone + " \n";
    
    llama_sampler_free(sampler);

    return histry;
}

string ssm(llama_model* model, const string& system_prompt, const string& user_prompt, int n_predict = 1024) {

    // создаём НОВЫЙ контекст (чистая память)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;

    llama_context* ctx = llama_new_context_with_model(model, cparams);

    // формируем промпт
    string full_prompt = "[INST] " + system_prompt +
        "\n\nПользователь: " + user_prompt +
        " [/INST]";

    // токенизация
    vector<llama_token> tokens = tokenize(ctx, full_prompt, true);

    // прогон промпта
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_free(ctx);
        return "Ошибка decode";
    }

    // greedy sampler (быстро и просто)
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    string result;

    for (int i = 0; i < n_predict; i++) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        const llama_vocab* vocab = llama_model_get_vocab(model);
        if (token == llama_token_eos(vocab)) break;

        string piece = token_to_piece(ctx, token);
        result += piece;

        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }

    llama_sampler_free(sampler);
    llama_free(ctx); // ВАЖНО — удаляем контекст

    return result;
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
    string sys_p = "";
    ifstream file2("C:/LLM_CPP/prompt.txt");
    if (file2.is_open()) {
        ostringstream ss;
        ss << file2.rdbuf(); // читаем весь файл
        sys_p = ss.str(); // всё содержимое тут
        file2.close();
        cout << "Промпт загрузилася" << endl;
    }

    else {
        sys_p = "Ты таисия - добрая ии подруга!";
        cout << "Файл не удалось загрущзить" << endl;
    }

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
    cparams.n_batch = 8192;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        cerr << "Ошибка создания контекста" << endl;
        llama_free_model(model);
        return 1;
    }

    string vospominaniy;
    ifstream file3("C:/LLM_CPP/history.txt");
    if (file3.is_open()) {
        ostringstream ss;
        ss << file3.rdbuf(); // читаем весь файл
        vospominaniy = ss.str(); // всё содержимое тут
        file3.close();
        cout << "Промпт загрузилася" << endl;
    }

    else {
        vospominaniy = "";
        cout << "Файл не удалось загрущзить" << endl;
    }
    cout << vospominaniy << endl;

    string chat_history = get_history("C:/LLM_CPP/history.jsonl");
    cout  << chat_history;
    for (;;) {
        cout << "____________________________________________________________________________" << endl;
        cout << "Ввод: ";
        string prompt;
        getline(cin, prompt);
        if (prompt == "выход" || prompt == ".") break;
        if (prompt.empty()) continue;
        if (prompt == "очистка истории") { clear_file("C:/LLM_CPP/history.jsonl"); continue; }
        if (prompt == "запусти ssm" || prompt == "SSM") {
            string system_prompt = "Ты память Таисии. Ты получаешь их переписку с Иваном, тебе нужно сократить переписку как ее карткое восопминание об этом времени. Кратко и поделу 2-3 предложения!";
            string user_prompt = chat_history;
            string response = ssm(model, system_prompt, user_prompt);
            cout << "Работа SSM: " << response << endl;
            chat_history = response;
            app_vospominania("C:/LLM_CPP/history.txt", chat_history);
        }
        else {
            chat_history += massag(prompt, vospominaniy, n_predict, model, ctx, sys_p);
        }
    }
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
