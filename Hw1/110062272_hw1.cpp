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
    TreeNode* link;
    
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

TreeNode* find(TreeNode* node, int id) {
    if (!node) return nullptr;
    for (auto child : node->children) {
        if (child->id == id) return child;
    }
    return nullptr;
}

bool cmp(int a, int b) {
    return counter[a] > counter[b];
}

vector<HeaderNode*> constructHeaderTable(int min_support_count) {
    vector<int> frequent_items;
    for (int i = 0; i < maxn; i++) {
        if (counter[i] >= min_support_count) {
            frequent_items.push_back(i);
        }
    }
    sort(frequent_items.begin(), frequent_items.end(), cmp);
    vector<HeaderNode*> headerTable;
    for (int id : frequent_items) {
        headerTable.push_back(new HeaderNode(id, counter[id]));
    }
    return headerTable;
}

void filterAndSortTransactions(vector<vector<int>>& transactions, const vector<HeaderNode*>& headerTable) {
    map<int, int> order;
    for (size_t i = 0; i < headerTable.size(); i++) {
        order[headerTable[i]->id] = i;
    }
    
    for (auto& transaction : transactions) {
        vector<int> filtered;
        for (int item : transaction) {
            if (counter[item] >= min_support_count) {
                filtered.push_back(item);
            }
        }
        
        sort(filtered.begin(), filtered.end(),[&order](int a, int b) { return order[a] < order[b]; });
        transaction = filtered;
    }
}


TreeNode* buildFPTree(vector<vector<int>>& transactions, vector<HeaderNode*>& headerTable) {
    TreeNode* root = new TreeNode(-1, 0); 
    map<int, HeaderNode*> headerMap;
    for (auto header : headerTable) {
        headerMap[header->id] = header;
    }
    
    for (auto& transaction : transactions) {
        TreeNode* curr = root;
        for (int item : transaction) {
            TreeNode* child = find(curr, item);
            
            if (child) {
                child->count++;
                curr = child;
            } else {
                TreeNode* newNode = new TreeNode(item, 1, curr);
                curr->children.push_back(newNode);
                
                HeaderNode* header = headerMap[item];
                if (!header->link) {
                    header->link = newNode;
                } else {
                    TreeNode* linkNode = header->link;
                    while (linkNode->link) {
                        linkNode = linkNode->link;
                    }
                    linkNode->link = newNode;
                }
                curr = newNode;
            }
        }
    }
    return root;
}

map<vector<int>, int> getConditionalPatternBase(HeaderNode* header) {
    map<vector<int>, int> patterns;
    TreeNode* node = header->link;
    
    while (node) {
        if (node->count > 0) {
            vector<int> path;
            TreeNode* parent = node->parent;
            
            while (parent && parent->id != -1) {
                path.push_back(parent->id);
                parent = parent->parent;
            }
            
            reverse(path.begin(), path.end());
            if (!path.empty()) {
                if (patterns.find(path) != patterns.end()) {
                    patterns[path] += node->count;
                } else {
                    patterns[path] = node->count;
                }
            }
        }
        node = node->link;
    }
    return patterns;
}

void fpGrowth(vector<HeaderNode*>& headerTable, vector<int>& prefix, double min_support_count) {
    for (int i = headerTable.size() - 1; i >= 0; i--) {
        HeaderNode* header = headerTable[i];
        vector<int> newPrefix = prefix;
        newPrefix.push_back(header->id);
        itemsetSupport[newPrefix] = (double)header->count / total_transactions;
        map<vector<int>, int> conditionalBase = getConditionalPatternBase(header);
        if (conditionalBase.empty()) {
            continue;
        }
        
        memset(counter, 0, sizeof(counter));
        
        for (const auto& pair : conditionalBase) {
            const vector<int>& path = pair.first;
            int count = pair.second;
            
            for (int item : path) {
                counter[item] += count;
            }
        }
        
        vector<HeaderNode*> conditionalHeaderTable = constructHeaderTable(min_support_count);
        
        if (conditionalHeaderTable.empty()) {
            continue;
        }
        
        vector<vector<int>> conditionalTransactions;
        for (const auto& pair : conditionalBase) {
            vector<int> transaction;
            const vector<int>& path = pair.first;
            
            for (int item : path) {
                if (counter[item] >= min_support_count) {
                    transaction.push_back(item);
                }
            }
            
            if (!transaction.empty()) {
                for (int j = 0; j < pair.second; j++) {
                    conditionalTransactions.push_back(transaction);
                }
            }
        }
        
        if (conditionalTransactions.empty()) {
            for (auto node : conditionalHeaderTable) {
                delete node;
            }
            continue;
        }
        
        filterAndSortTransactions(conditionalTransactions, conditionalHeaderTable);
        
        TreeNode* conditionalTree = buildFPTree(conditionalTransactions, conditionalHeaderTable);
        
        fpGrowth(conditionalHeaderTable, newPrefix, min_support_count);
        
        delete conditionalTree;
        for (auto node : conditionalHeaderTable) {
            delete node;
        }
    }
}

int main(int argc, char *argv[]) {
    rein
    ifstream inFile;
    ofstream outFile;
    
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <min_support_ratio> <input_file> <output_file>" << endl;
        return 1;
    }
    
    min_support_ratio = atof(argv[1]);
    string inputFileName = argv[2];
    string outputFileName = argv[3];

    inFile.open(inputFileName);
    if (!inFile) {
        cerr << "Unable to open file " << inputFileName << endl;
        exit(1); 
    }

    string line;
    stringstream ss;
    vector<vector<int>> transactions;
    
    memset(counter, 0, sizeof(counter));
    
    while (getline(inFile, line)) {
        ss.str("");
        ss.clear();
        ss << line;
        
        vector<int> transaction;
        string item;
        
        while (getline(ss, item, ',')) {
            int id = stoi(item);
            transaction.push_back(id);
            counter[id]++;
        }
        
        transactions.push_back(transaction);
    }
    inFile.close();
    
    total_transactions = transactions.size();
    min_support_count = min_support_ratio * total_transactions;
    
    // cout << "Total transactions: " << total_transactions << endl;
    // cout << "Min support count: " << min_support_count << endl;
    
    vector<HeaderNode*> headerTable = constructHeaderTable(min_support_count);
    
    filterAndSortTransactions(transactions, headerTable);
    
    TreeNode* fpTree = buildFPTree(transactions, headerTable);

    for (auto header : headerTable) {
        vector<int> singleItem = {header->id};
        double support = (double)header->count / total_transactions;
        itemsetSupport[singleItem] = support;
    }
    
    vector<int> prefix;
    fpGrowth(headerTable, prefix, min_support_count);

    outFile.open(outputFileName);
    if (!outFile) {
        cerr << "Unable to open output file " << outputFileName << endl;
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
        outFile << "\n";
    }
    
    // cout << "Found " << itemsetSupport.size() << " frequent itemsets." << endl;
    outFile.close();
    
    delete fpTree;
    for (auto header : headerTable) {
        delete header;
    }
    
    return 0;
}