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

struct VectorCompare {
    bool operator()(const vector<int>& a, const vector<int>& b) const {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        return a < b;
    }
};

map<vector<int>, double, VectorCompare> itemsetSupport;
double min_support_ratio;
double min_support_count;
int total_transactions;

class TreeNode {
public:
    int id;
    int count;
    TreeNode* parent;
    vector<TreeNode*> children;
    TreeNode* link;  // 用於headerTable的鏈接
    
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

class HeaderNode {
public:
    int id;
    int count;
    TreeNode* link;
    
    HeaderNode(int id, int count, TreeNode* link=NULL) {
        this->id = id; 
        this->count = count;
        this->link = link;
    }
};

// 函數聲明
TreeNode* find(TreeNode* node, int id);
bool cmp(int a, int b);
void filterTransactions(vector<vector<int>>& transactions);
TreeNode* buildTree(vector<HeaderNode*>& headerTable, vector<vector<int>>& transactions);
void minePatterns(TreeNode* tree, vector<int>& prefix, double min_support_count);
vector<HeaderNode*> constructHeaderTable(vector<vector<int>>& transactions, int min_support_count);

// 函數實現
TreeNode* find(TreeNode* node, int id) {
    if (!node) return nullptr;
    for (auto child : node->children) {
        if (child->id == id) return child;
    }
    return nullptr;
}

int getItemSupport(TreeNode* node) {
    int support = 0;
    while (node) {
        support += node->count;
        node = node->link;
    }
    return support;
}

bool cmp(int a, int b) {
    return counter[a] > counter[b];
}

vector<HeaderNode*> constructHeaderTable(vector<vector<int>>& transactions, int min_support_count) {
    vector<HeaderNode*> headerTable;
    
    vector<int> filtered;
    for (int i = 0; i < maxn; i++) {
        if (counter[i] >= min_support_count) {
            filtered.push_back(i);
        }
    }
    
    sort(filtered.begin(), filtered.end(), cmp);
    
    for (int id : filtered) {
        headerTable.push_back(new HeaderNode(id, counter[id]));
    }
    return headerTable;
}

void filterTransactions(vector<vector<int>>& transactions) {
    for(auto& transaction : transactions) {
        vector<int> filteredTransaction;
        for (int item : transaction) {
            if (counter[item] >= min_support_count) {
                filteredTransaction.push_back(item);
            }
        }
        // 按頻率排序
        sort(filteredTransaction.begin(), filteredTransaction.end(), cmp);
        transaction = filteredTransaction;
    }
}

TreeNode* buildTree(vector<HeaderNode*>& headerTable, vector<vector<int>>& transactions) {
    TreeNode* root = new TreeNode(-1, 0);
    for(auto transaction : transactions) {
        TreeNode* current = root;
        for(int item : transaction) {
            TreeNode* child = find(current, item);
            if(child == NULL) {
                TreeNode* newNode = new TreeNode(item, 1, current, nullptr);
                current->children.push_back(newNode);
                for (HeaderNode* header : headerTable) {
                    if (header->id == item) {
                        if (header->link == nullptr) {
                            header->link = newNode;
                        } else {
                            TreeNode* temp = header->link;
                            while (temp->link != nullptr) {
                                temp = temp->link;
                            }
                            temp->link = newNode;
                        }
                        break;
                    }
                }
                current = newNode; 
            }
            else {
                child->count++;
                current = child;
            }
        }
    }
    return root; 
}

vector<vector<int>> getConditionalPatternBase(TreeNode* node) {
    vector<vector<int>> patterns;
    
    while (node != nullptr) {
        // 從節點向上追溯到根（除了根節點）
        vector<int> pattern;
        TreeNode* parent = node->parent;
        while (parent != nullptr && parent->id != -1) {
            pattern.push_back(parent->id);
            parent = parent->parent;
        }
        
        // 如果路徑不為空，將其複製node->count次添加到條件模式基中
        if (!pattern.empty()) {
            reverse(pattern.begin(), pattern.end()); // 順序應該是從根到葉
            for (int i = 0; i < node->count; i++) {
                patterns.push_back(pattern);
            }
        }
        
        node = node->link;
    }
    
    return patterns;
}

void mineTree(vector<HeaderNode*>& headerTable, vector<int>& prefix, double min_support_count) {
    // 按頻率從小到大處理條目（這樣可以更快地到達簡短的前綴）
    for (int i = headerTable.size() - 1; i >= 0; i--) {
        HeaderNode* header = headerTable[i];
        
        // 生成新的頻繁項集前綴
        vector<int> newPrefix = prefix;
        newPrefix.push_back(header->id);
        
        // 將新的頻繁項集添加到結果中
        double support = header->count / (double)total_transactions;
        itemsetSupport[newPrefix] = support;
        
        // 構建條件模式基
        vector<vector<int>> conditionalPatterns = getConditionalPatternBase(header->link);
        
        if (conditionalPatterns.empty()) continue;
        
        // 重置計數器
        memset(counter, 0, sizeof(counter));
        
        // 計算條件模式基中每個項的支持度
        for (auto& pattern : conditionalPatterns) {
            for (int item : pattern) {
                counter[item]++;
            }
        }
        
        // 構建條件FP樹的頭表
        vector<HeaderNode*> conditionalHeaderTable;
        for (int j = 0; j < maxn; j++) {
            if (counter[j] >= min_support_count) {
                conditionalHeaderTable.push_back(new HeaderNode(j, counter[j]));
            }
        }
        
        if (conditionalHeaderTable.empty()) continue;
        
        // 按頻率排序
        sort(conditionalHeaderTable.begin(), conditionalHeaderTable.end(), 
            [](HeaderNode* a, HeaderNode* b) { return counter[a->id] > counter[b->id]; });
        
        // 過濾條件模式基
        filterTransactions(conditionalPatterns);
        
        // 構建條件FP樹
        TreeNode* conditionalTree = buildTree(conditionalHeaderTable, conditionalPatterns);
        
        // 遞歸挖掘條件FP樹
        mineTree(conditionalHeaderTable, newPrefix, min_support_count);
        
        // 清理
        delete conditionalTree;
        for (auto node : conditionalHeaderTable) {
            delete node;
        }
    }
}

void minePatterns(TreeNode* tree, vector<int>& prefix, double min_support_count) {
    // 如果樹只有根節點，直接返回
    if (tree->children.empty()) return;
    
    // 構建頭表
    vector<HeaderNode*> headerTable;
    for (auto child : tree->children) {
        // 遍歷所有分支收集項集
        int itemId = child->id;
        int support = 0;
        
        // 計算每個項的支持度
        TreeNode* node = child;
        while (node != nullptr) {
            support += node->count;
            node = node->link;
        }
        
        // 如果支持度足夠，加入頭表
        if (support >= min_support_count) {
            headerTable.push_back(new HeaderNode(itemId, support));
        }
    }
    
    // 遞歸挖掘
    mineTree(headerTable, prefix, min_support_count);
    
    // 清理頭表
    for (auto node : headerTable) {
        delete node;
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
    vector<int> transaction;
    int num = 0;

    memset(counter, 0, sizeof(counter));

    while (getline(inFile, line)) {
        ss.str("");
        ss.clear();
        ss << line;
        string item;
        transaction.clear();
        while (getline(ss, item, ',')) {
            int id = stoi(item);
            transaction.push_back(id);
            counter[id]++;
        }
        transactions.push_back(transaction);
        num++;
    }
    inFile.close();
    
    total_transactions = transactions.size();
    min_support_count = min_support_ratio * total_transactions;
    
    cout << "Total transactions: " << total_transactions << endl;
    cout << "Min support count: " << min_support_count << endl;

    // 構建頭表
    vector<HeaderNode*> headerTable = constructHeaderTable(transactions, min_support_count);
    
    // 過濾和排序交易記錄
    filterTransactions(transactions);
    
    // 構建FP樹
    TreeNode* root = buildTree(headerTable, transactions);
    
    // 挖掘頻繁項集
    vector<int> prefix;
    minePatterns(root, prefix, min_support_count);

    // 輸出結果
    outFile.open(outputFileName);
    if (!outFile) {
        cerr << "Unable to open file " << outputFileName << endl;
        exit(1);
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
    for (auto node : headerTable) {
        delete node;
    }
    delete root;
    
    return 0;
}