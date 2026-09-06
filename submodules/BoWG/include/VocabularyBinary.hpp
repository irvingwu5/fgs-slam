// 文件作用：序列化和反序列化视觉词典；隶属于 词典二进制存取 模块。
#ifndef VocabularyBinary_hpp
#define VocabularyBinary_hpp

#include <cstdint>
#include <fstream>
#include <string>

namespace VINSLoop {
    
struct Node {
    int32_t nodeId;
    int32_t parentId;
    double weight;
    uint64_t descriptor[4];
};

struct Word {
    int32_t nodeId;
    int32_t wordId;
};

struct Vocabulary {
    int32_t k;
    int32_t L;
    int32_t scoringType;
    int32_t weightingType;
    
    int32_t nNodes;
    int32_t nWords;
    
    Node* nodes;
    Word* words;
    
// 函数作用：执行 Vocabulary 对应的构造、计算或状态操作。
// 所属模块：词典二进制存取。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    Vocabulary();
// 函数作用：释放对象持有的资源。
// 所属模块：词典二进制存取。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    ~Vocabulary();
    
// 函数作用：执行 serialize 对应的构造、计算或状态操作。
// 所属模块：词典二进制存取。
// 输入：
//   - stream：输入或输出数据流。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void serialize(std::ofstream& stream);
// 函数作用：执行 deserialize 对应的构造、计算或状态操作。
// 所属模块：词典二进制存取。
// 输入：
//   - stream：输入或输出数据流。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void deserialize(std::ifstream& stream);
    
// 函数作用：执行 staticDataSize 对应的构造、计算或状态操作。
// 所属模块：词典二进制存取。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    inline static size_t staticDataSize() {
        return sizeof(Vocabulary) - sizeof(Node*) - sizeof(Word*);
    }
};

}

#endif /* VocabularyBinary_hpp */
