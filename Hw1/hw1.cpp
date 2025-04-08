#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cmath>
#include <cstring>
using namespace std;
const int maxn = 1000000;
int counter[maxn];
#define rein ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);

// 比較向量，用於map的排序
struct VectorCompare {
    bool operator()(const vector<int>& a, const vector<int>& b) const {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        return a < b;
    }
};

// 用於存儲頻繁項集及其支持度
map<vector<int>, double, VectorCompare> itemsetSupport;
double min_support_ratio;
double min_support_count;
int total_transactions;

// FP樹的節點
class TreeNode {
public:
    int id;
    int count;
    TreeNode* parent;
    vector<TreeNode*> children;
    TreeNode* link;  // 指向相同項的下一個節點
    
    TreeNode(int id, int count, TreeNode *parent=NULL, TreeNode *link=NULL) {
        this->id = id;
        this->count = count;
        this->parent = parent;
        this->link = link;
    }

    ~TreeNode() {
        for (auto child : children) {
            delete child;
        }
    }
};

// 頭表的節點
class HeaderNode {
public:
    int id;
    int count;
    TreeNode* link;  // 指向FP樹中相應項的第一個節點
    
    HeaderNode(int id, int count, TreeNode* link=NULL) {
        this->id = id; 
        this->count = count;
        this->link = link;
    }
};

// 在節點的子節點中查找特定ID的節點
TreeNode* find(TreeNode* node, int id) {
    if (!node) return nullptr;
    for (auto child : node->children) {
        if (child->id == id) return child;
    }
    return nullptr;
}

// 比較函數：按支持度降序排序
bool cmp(int a, int b) {
    return counter[a] > counter[b];
}

// 構建頭表
vector<HeaderNode*> constructHeaderTable(vector<vector<int>>& transactions, double min_support_count) {
    memset(counter, 0, sizeof(counter));
    
    // 計算每個項的支持度
    for (auto& transaction : transactions) {
        for (int item : transaction) {
            counter[item]++;
        }
    }
    
    // 收集頻繁項
    vector<int> frequent_items;
    for (int i = 0; i < maxn; i++) {
        if (counter[i] >= min_support_count) {
            frequent_items.push_back(i);
        }
    }
    
    // 按支持度排序
    sort(frequent_items.begin(), frequent_items.end(), cmp);
    
    // 創建頭表
    vector<HeaderNode*> headerTable;
    for (int id : frequent_items) {
        headerTable.push_back(new HeaderNode(id, counter[id]));
    }
    
    return headerTable;
}

// 過濾交易並按支持度排序
void filterAndSortTransactions(vector<vector<int>>& transactions, vector<HeaderNode*>& headerTable) {
    // 創建項ID到排序權重的映射
    map<int, int> order;
    for (size_t i = 0; i < headerTable.size(); i++) {
        order[headerTable[i]->id] = i;
    }
    
    for (auto& transaction : transactions) {
        // 過濾不頻繁的項
        vector<int> filtered;
        for (int item : transaction) {
            if (counter[item] >= min_support_count) {
                filtered.push_back(item);
            }
        }
        
        // 按頭表順序排序
        sort(filtered.begin(), filtered.end(), 
             [&order](int a, int b) { return order[a] < order[b]; });
        
        transaction = filtered;
    }
}

// 構建FP樹
TreeNode* buildFPTree(vector<vector<int>>& transactions, vector<HeaderNode*>& headerTable) {
    TreeNode* root = new TreeNode(-1, 0);  // 根節點
    
    // 創建項ID到頭表節點的映射，方便快速查找
    map<int, HeaderNode*> headerMap;
    for (auto header : headerTable) {
        headerMap[header->id] = header;
    }
    
    // 將每筆交易添加到FP樹
    for (auto& transaction : transactions) {
        TreeNode* curr = root;
        
        for (int item : transaction) {
            // 查找當前節點是否已有此項的子節點
            TreeNode* child = find(curr, item);
            
            if (child) {
                // 已存在，增加計數
                child->count++;
                curr = child;
            } else {
                // 不存在，創建新節點
                TreeNode* newNode = new TreeNode(item, 1, curr);
                curr->children.push_back(newNode);
                
                // 更新頭表的鏈接
                HeaderNode* header = headerMap[item];
                if (!header->link) {
                    header->link = newNode;
                } else {
                    // 找到鏈的末尾
                    TreeNode* linkNode = header->link;
                    while (linkNode->link) {
                        linkNode = linkNode->link;
                    }
                    linkNode->link = newNode;
                }
                
                // 更新當前節點
                curr = newNode;
            }
        }
    }
    
    return root;
}

// 從一個節點獲取其條件模式基
vector<pair<vector<int>, int>> getConditionalPatternBase(HeaderNode* header) {
    vector<pair<vector<int>, int>> patterns;
    TreeNode* node = header->link;
    
    while (node) {
        // 收集從當前節點到根的路徑（不包括根和當前項）
        vector<int> path;
        TreeNode* parent = node->parent;
        
        while (parent && parent->id != -1) {
            path.push_back(parent->id);
            parent = parent->parent;
        }
        
        if (!path.empty()) {
            reverse(path.begin(), path.end());  // 路徑應該從根到葉
            patterns.push_back({path, node->count});
        }
        
        node = node->link;  // 移動到下一個相同的項
    }
    
    return patterns;
}

// 遞歸挖掘頻繁項集
void fpGrowth(vector<HeaderNode*>& headerTable, vector<int>& prefix, double min_support_count) {
    // 對每個頭表項處理（從最不頻繁的開始）
    for (int i = headerTable.size() - 1; i >= 0; i--) {
        HeaderNode* header = headerTable[i];
        
        // 創建新的前綴
        vector<int> newPrefix = prefix;
        newPrefix.push_back(header->id);
        
        // 添加到頻繁項集
        itemsetSupport[newPrefix] = (double)header->count / total_transactions;
        
        // 獲取條件模式基
        vector<pair<vector<int>, int>> conditionalBase = getConditionalPatternBase(header);
        
        if (conditionalBase.empty()) {
            continue;
        }
        
        // 創建條件FP樹的交易數據
        vector<vector<int>> conditionalTransactions;
        for (auto& pattern : conditionalBase) {
            vector<int>& path = pattern.first;
            int count = pattern.second;
            
            for (int j = 0; j < count; j++) {
                conditionalTransactions.push_back(path);
            }
        }
        
        // 構建條件FP樹的頭表
        vector<HeaderNode*> conditionalHeaderTable = constructHeaderTable(conditionalTransactions, min_support_count);
        
        if (conditionalHeaderTable.empty()) {
            continue;
        }
        
        // 過濾和排序條件交易
        filterAndSortTransactions(conditionalTransactions, conditionalHeaderTable);
        
        // 構建條件FP樹
        if (!conditionalTransactions.empty()) {
            // 遞歸處理
            fpGrowth(conditionalHeaderTable, newPrefix, min_support_count);
        }
        
        // 清理
        for (auto node : conditionalHeaderTable) {
            delete node;
        }
    }
}

int main(int argc, char *argv[]) {
    ifstream inFile;
    ofstream outFile;
    
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <min_support_ratio> <input_file> <output_file>" << endl;
        return 1;
    }
    
    min_support_ratio = atof(argv[1]);
    string inputFileName = argv[2];
    string outputFileName = argv[3];

    cout << "min_support: " << min_support_ratio << endl;
    cout << "inputFileName: " << inputFileName << endl;
    cout << "outputFileName: " << outputFileName << endl;

    inFile.open(inputFileName);
    if (!inFile) {
        cerr << "Unable to open file " << inputFileName << endl;
        exit(1); 
    }

    string line;
    stringstream ss;
    vector<vector<int>> transactions;
    
    // 讀取交易數據
    while (getline(inFile, line)) {
        ss.str("");
        ss.clear();
        ss << line;
        
        vector<int> transaction;
        string item;
        
        while (getline(ss, item, ',')) {
            transaction.push_back(stoi(item));
        }
        
        transactions.push_back(transaction);
    }
    inFile.close();
    
    total_transactions = transactions.size();
    min_support_count = min_support_ratio * total_transactions;
    
    cout << "Total transactions: " << total_transactions << endl;
    cout << "Min support count: " << min_support_count << endl;
    
    // 構建頭表
    vector<HeaderNode*> headerTable = constructHeaderTable(transactions, min_support_count);
    
    if (headerTable.empty()) {
        cout << "No frequent items found." << endl;
        return 0;
    }
    
    // 過濾和排序交易
    filterAndSortTransactions(transactions, headerTable);
    
    // 構建FP樹
    TreeNode* fpTree = buildFPTree(transactions, headerTable);
    
    // 挖掘頻繁項集
    vector<int> emptyPrefix;
    fpGrowth(headerTable, emptyPrefix, min_support_count);
    
    // 輸出結果
    outFile.open(outputFileName);
    if (!outFile) {
        cerr << "Unable to open output file " << outputFileName << endl;
        exit(1);
    }
    
    // 輸出單項頻繁集
    for (auto header : headerTable) {
        vector<int> singleItem = {header->id};
        double support = (double)header->count / total_transactions;
        itemsetSupport[singleItem] = support;
    }
    
    for (const auto& pair : itemsetSupport) {
        const vector<int>& itemset = pair.first;
        double support = pair.second;
        
        for (size_t i = 0; i < itemset.size(); i++) {
            outFile << itemset[i];
            if (i < itemset.size() - 1) {
                outFile << ",";
            }
        }
        
        outFile << ":" << fixed << setprecision(4) << support;
        outFile << endl;
    }
    
    cout << "Found " << itemsetSupport.size() << " frequent itemsets." << endl;
    outFile.close();
    
    // 清理內存
    delete fpTree;
    for (auto header : headerTable) {
        delete header;
    }
    
    return 0;
}