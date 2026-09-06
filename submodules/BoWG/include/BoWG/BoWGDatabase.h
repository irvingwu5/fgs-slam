// 文件作用：实现视觉词量化、倒排数据库、词组评分和回环检测；隶属于 BoW/BoWG 检索 模块。
/**
 * File: BoWGDatabase.h
 * Date: August 2024
 * Author: Xiang Fei
 * Description: DoWG database of images
 *
 */

#ifndef __BOWG_DATABASE__
#define __BOWG_DATABASE__

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "DBoW2.h"
#include "TemplatedDatabase.h"
#include "TemplatedVocabulary.h"

#include "BoWGVector.h"
#include "BoWGScoring.h"

namespace BoWG {

// Id of items of the database (image id)
typedef unsigned int ItemID;

class BoWGDatabase
{
public:

    /**
    * Constructor of BoWGDatabase
    * @param wg_weight_type weighting type of word group
    * @param wg_scoring_type scoring type of word group
    * @param w_scoring_type scoring type of word (for computing the normalized score)
    */
// 函数作用：执行 BoWGDatabase 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - wg_weight_type：权重或加权方式。
//   - wg_scoring_type：wg_scoring_type 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - w_scoring_type：w_scoring_type 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    BoWGDatabase(int wg_weight_type = 0, int wg_scoring_type = 1, int w_scoring_type = 0);

    /**
    * Destructor of BoWGDatabase
    */
// 函数作用：释放对象持有的资源。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：无返回值；结果通过对象状态、输出参数、文件或控制台产生。
    ~BoWGDatabase(void);

    /**
    * check whether the current word groups is in the database or not
    * if yes, return the word group id, otherwise, add the word group into our database and assign its id
    * @param wid1 center word id of the consider word groups
    */ 
// 函数作用：查询 queryWordGroup 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - wid1：节点、词或条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    WordGroupID queryWordGroup(DBoW2::WordId wid1);

    /**
    * add new world group to the database and return its id
    * @param wg_keys center word id of the consider word groups
    */ 
// 函数作用：执行 db_add 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - wg_key：wg_key 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    WordGroupID db_add(DBoW2::WordId wg_key);

    /**
    * obtain the BoWGVector for an image
    * @param wg_ids word group ids of an image
    */ 
// 函数作用：由局部词组统计构造并归一化 BoWG 向量。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - wg_ids：节点、词或条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    BoWGVector computeBoWGVector(std::vector<WordGroupID> wg_ids);

    /**
    * update the BoWGVector Table and Inverse Index Table
    * @param bowgVector BoWGVector of the image
    * @param entry_id Id of the image
    */ 
// 函数作用：向数据库添加 add 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - bowgVector：bowgVector 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - entry_id：节点、词或条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void add(BoWGVector& bowgVector, ItemID entry_id);

    /**
    * update the Distribution Table
    * @param vec distribution vector of the image
    */ 
// 函数作用：执行 dist_add 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void dist_add(std::vector<int>& vec);

    // obtain the image id
// 函数作用：获取 get_itemId 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    ItemID get_itemId(void);

    /**
    * query the word group results using inverse index table
    * @param vec BoWGVector of the query image
    * @param ret original word group query results
    * @param wg_item_vec a vector for later word/word group results alignment
    * @param max_results max number of results
    * @param max_id the max id of images
    */ 
// 函数作用：执行 wg_queryL1 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - ret：查询结果输出容器。
//   - wg_item_vec：wg_item_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void wg_queryL1(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
// 函数作用：执行 wg_queryL2 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - ret：查询结果输出容器。
//   - wg_item_vec：wg_item_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void wg_queryL2(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
// 函数作用：执行 wg_queryCHI_SQUARE 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - ret：查询结果输出容器。
//   - wg_item_vec：wg_item_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void wg_queryCHI_SQUARE(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
// 函数作用：执行 wg_queryKL 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - ret：查询结果输出容器。
//   - wg_item_vec：wg_item_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void wg_queryKL(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
// 函数作用：执行 wg_queryBHATTACHARYYA 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - ret：查询结果输出容器。
//   - wg_item_vec：wg_item_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void wg_queryBHATTACHARYYA(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);
// 函数作用：执行 wg_queryDOT 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - vec：vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - ret：查询结果输出容器。
//   - wg_item_vec：wg_item_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void wg_queryDOT(const BoWGVector &vec, DBoW2::QueryResults &ret, std::vector<ItemID> &wg_item_vec, int max_results = 1, int max_id = -1);

    /**
    * query the final results using normalized and temporal word score
    * @param ret original word query results
    * @param out_ret output results
    * @param bowVector bowVector of the query image
    * @param max_results max number of results
    * @param use_temporal_score whether to use temporal score
    * @param prev_weight_th the maximum previou score weight
    * @param temporal_param temporal score parameter to compute the weight
    */ 
// 函数作用：执行词级倒排检索与评分。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
//   - out_ret：out_ret 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - bowVector：bowVector 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - use_temporal_score：相似度分数或评分对象。
//   - prev_weight_th：权重或加权方式。
//   - temporal_param：temporal_param 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void query_words(DBoW2::QueryResults &ret, DBoW2::QueryResults &out_ret, DBoW2::BowVector &bowVector, int max_results, 
                        bool use_temporal_score=true, double prev_weight_th = 0.5, double temporal_param=1.0);

    /**
    * query the final results using normalized and temporal word score and word group score
    * @param ret original word query results
    * @param out_ret output results
    * @param bowVector bowVector of the query image
    * @param bowgVector bowgVector of the query image
    * @param max_results max number of results
    * @param max_id the max id of images
    * @param w_weight the weight of word score for combined score
    * @param use_temporal_score whether to use temporal score
    * @param prev_weight_th the maximum previou score weight
    * @param temporal_param temporal score parameter to compute the weight
    */ 
// 函数作用：融合词级与组级分数并返回候选。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
//   - out_ret：out_ret 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - bowVector：bowVector 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - bowgVector：bowgVector 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
//   - w_weight：权重或加权方式。
//   - use_temporal_score：相似度分数或评分对象。
//   - prev_weight_th：权重或加权方式。
//   - temporal_param：temporal_param 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void query_bowg(DBoW2::QueryResults &ret, DBoW2::QueryResults &out_ret, DBoW2::BowVector &bowVector, BoWGVector &bowgVector, int max_results = 1, 
            int max_id = -1, double w_weight=0.7, bool use_temporal_score=true, double prev_weight_th = 0.5, double temporal_param=1.0);

    
    /**
    * query the final results using normalized and temporal word score, word group score, and distribution score
    * @param ret original word query results
    * @param out_ret output results
    * @param bowVector bowVector of the query image
    * @param bowgVector bowgVector of the query image
    * @param dist_vec distribution vector of the query image
    * @param max_results max number of results
    * @param max_id the max id of images
    * @param w_weight the weight of word score for combined score
    * @param wg_weight the weight of word group score for combined score
    * @param use_temporal_score whether to use temporal score
    * @param prev_weight_th the maximum previou score weight
    * @param temporal_param temporal score parameter to compute the weight
    */ 
// 函数作用：融合词级与组级分数并返回候选。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
//   - out_ret：out_ret 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - bowVector：bowVector 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - bowgVector：bowgVector 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - dist_vec：dist_vec 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_results：最多返回的候选数量。
//   - max_id：允许检索的最大历史条目 ID。
//   - w_weight：权重或加权方式。
//   - wg_weight：权重或加权方式。
//   - use_temporal_score：相似度分数或评分对象。
//   - prev_weight_th：权重或加权方式。
//   - temporal_param：temporal_param 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void query_bowg(DBoW2::QueryResults &ret, DBoW2::QueryResults &out_ret, DBoW2::BowVector &bowVector, BoWGVector &bowgVector, std::vector<int> dist_vec, int max_results = 1, 
        int max_id = -1, double w_weight=0.35, double wg_weight=0.35, bool use_temporal_score=true, double prev_weight_th = 0.5, double temporal_param=1.0);

    /**
    * compute the islands
    * @param ret query results
    * @param groupMatches islands
    * @param candidates_th similarity threshold
    * @param max_intraisland_gap max separation between matches to consider them of the same island
    */ 
// 函数作用：匹配 matchGrouping 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
//   - groupMatches：groupMatches 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - candidates_th：节点、词或条目 ID。
//   - max_intraisland_gap：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    bool matchGrouping(DBoW2::QueryResults &ret, std::vector<std::vector<int>> &groupMatches, double candidates_th, int max_intraisland_gap);

    // obtain the matched island index
// 函数作用：执行 islandMatching 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
//   - groupMatches：groupMatches 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    int islandMatching(DBoW2::QueryResults &ret, std::vector<std::vector<int>> &groupMatches);

    // temporal consistency check, consider previous k islands
// 函数作用：执行 temporalConsistency 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
//   - island：island 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - k：k 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - max_group_gap：上限参数。
//   - max_query_gap：上限参数。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    bool temporalConsistency(DBoW2::QueryResults &ret, std::vector<int> &island, int k, int max_group_gap, int max_query_gap);

    // get the number of word groups
// 函数作用：获取 get_num_wg 对应的数据或状态。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    int get_num_wg(void);

    /*Compute the input of kernel density estimation
     *input is the keypoints of the current keyframe */ 
// 函数作用：执行 kernel_input 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - keypoints：图像关键点集合。
//   - center_pt：center_pt 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
//   - batch：batch 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    std::vector<int> kernel_input(std::vector<cv::KeyPoint> keypoints, cv::Point2f center_pt, int batch);

    // stores the last time query result
// 函数作用：执行 store_last_res 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - ret：查询结果输出容器。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
    void store_last_res(DBoW2::QueryResults &ret);

    // last time query result, used to compute our proposed temporal consistency score
    std::map<int, double> last_query_res;

    // stores the information of matched islands in timesteps
    std::map<int, std::vector<int>> Islands_map;

    // stores the query results in timesteps
    std::map<int, DBoW2::QueryResults> res_table;

    // previous bow vector, used for normalization
    DBoW2::BowVector prev_bow_vec;

    // previous bowg vector, used for normalization
    BoWGVector prev_bowg_vec;

    // previous bowg vector, used for normalization
    std::vector<int> prev_dist_vec;

    // Scoring
    BoWGScoring bowg_scoring;

    int cur_image_id = 0;

    double min_prev_score = 0.005;
    double min_prev_wg_score = 0.005;
    double min_prev_dist_score = 0.003;

protected:

    /// Item of IFRow
    struct GroupIFPair
    {
        /// Item id
        ItemID item_id;
        
        /// Word group weight in this item
        WordGroupValue wg_weight;
        
        /**
        * Creates an empty pair
        */
// 函数作用：执行 GroupIFPair 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：无。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
        GroupIFPair(){}
        
        /**
        * Creates an inverted file pair
        * @param eid item id
        * @param wv word group weight
        */
// 函数作用：执行 GroupIFPair 对应的构造、计算或状态操作。
// 所属模块：BoW/BoWG 检索。
// 输入：
//   - eid：节点、词或条目 ID。
//   - wv：wv 所表示的计算输入；具体类型和约束由函数签名及调用上下文确定。
// 输出：返回函数声明类型规定的计算结果；若为 void，则通过对象状态或输出参数产生结果。
        GroupIFPair(ItemID eid, WordGroupValue wv): item_id(eid), wg_weight(wv) {}
        
        /**
        * Compares the item ids
        * @param eid
        * @return true iff this item id is the same as eid
        */
        inline bool operator==(ItemID eid) const { return item_id == eid; }
    };

    /// Row of InvertedFile
    typedef std::list<GroupIFPair> GroupIFRow;
    // GroupIFRow are sorted in ascending item id order

    /// Inverted index
    typedef std::map<WordGroupID,GroupIFRow> GroupInvertedFile; 
    // GroupInvertedFile[wg_id] --> inverted file of that word group

    /* BoWGVector table declaration*/
    typedef std::vector<BoWGVector> BoWGVectorTable;
    // BoWGVectorTable[item_id] --> the BoWGVector of an image, only useful when don't want to use inverse index table to compute scores

    // Distribution table decalaration
    typedef std::vector<std::vector<int>> DisTable;
    // DisTable[item_id] --> distribution vector of an image

    // IDF Table
    typedef std::map<WordGroupID,int> IDFTable;

protected:
    // vocabulary (BoWG Table) wg id and the number of the wg
    std::map<WordGroupID, int> voc_wg;

    // inverted index table
    GroupInvertedFile m_ifile;

    // BoWGVector table
    BoWGVectorTable m_dtable;

    // distribution table
    DisTable dis_table;

    // total word groups (consider repeat)
    int numOfWg;

    int w_scoring_type; // to compute the word normalization score
    int wg_weight_type;
    int wg_scoring_type;
};

} // namespace BoWG


#endif