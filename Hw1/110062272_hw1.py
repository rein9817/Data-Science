import argparse 
import sys
import time

class FPNode:
    def __init__(self,item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

class FPtree:
    def __init__(self):
        self.root = FPNode()
        self.header_table = {}
        self.min_support = 0
    
    def update_header_table(self, node):
        if node.item in self.header_table:
            current_node = self.header_table[node.item]
            while current_node.next:
                current_node = current_node.next
            current_node.next = node
        else:
            self.header_table[node.item] = node
            
    def add_transaction(self, transaction, count=1):
        current = self.root
        for item in transaction:
            if item in current.children:
                current.children[item].count += count
            else:
                new_node = FPNode(item, count, current)
                current.children[item] = new_node
                self.update_header_table(new_node)
                
            current = current.children[item]
            
    
    def find_prefix_path(self, node):
        paths = []
        while node.parent and node.parent.item is not None:
            paths.append(node.parent.item)
            node = node.parent
        
        return list(reversed(paths)) if paths else []
    
    def find_all_paths(self, item):
        paths = []
        node = self.header_table[item]
        
        while node:
            prefix_path = self.find_prefix_path(node)
            if prefix_path:
                paths.append(prefix_path)
            node = node.next
        return paths

def preprocessing(transactions):
    item_count = {}
    for transaction in transactions:
        for item in transaction:
            item_count[item] = item_count.get(item,0) + 1
    return item_count

def build_tree(transactions, min_support_count):
    item_count = preprocessing(transactions)
    frequent_items = {
        item: count for item, count in item_count.items() if count >= min_support_count
    }
    
    if not frequent_items:
        return None,None
    
    fp_tree = FPtree()
    fp_tree.min_support = min_support_count
    
    for transaction in transactions:
        filtered_transaction = [item for item in transaction if item in frequent_items]
        if not filtered_transaction:
            continue
        
        filtered_transaction.sort(key=lambda x: item_count[x], reverse=True)
        fp_tree.add_transaction(filtered_transaction)
        
    return fp_tree,frequent_items

def data_mining(fp_tree, prefix, frequent_patterns, min_support_count):
    items = list(fp_tree.header_table.keys())
    
    for item in items:
        new_pattern = prefix.copy()
        new_pattern.append(item)
        
        support = 0
        node = fp_tree.header_table[item]
        while node:
            support += node.count
            node = node.next
        
        if support >= min_support_count:
            frequent_patterns[tuple(sorted(new_pattern))] = support
            conditional_pattern = []
            node = fp_tree.header_table[item]
            
            while node:
                path = fp_tree.find_prefix_path(node)
                if path:
                    conditional_pattern.append((path , node.count))
                node = node.next
            
            conditional_tree = FPtree()
            for path , count in conditional_pattern:
                conditional_tree.add_transaction(path, count)   
            
            if conditional_tree.header_table:
                data_mining(conditional_tree, new_pattern, frequent_patterns, min_support_count)
                    
def fp_growth(transactions, min_support):
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    
    fp_tree, frequent_items = build_tree(transactions, min_support_count)
    if not fp_tree:
        return {}
    frequent_patterns = {(item,): count for item, count in frequent_items.items()}
    data_mining(fp_tree, [], frequent_patterns, min_support_count)
    
    return {pattern: count / num_transactions for pattern, count in frequent_patterns.items()}

def format_output(patterns):
    output = []
    for pattern,support in patterns.items():
        pattern_str = ','.join(map(str,pattern))
        output.append(f"{pattern_str}:{support:.4f}")
    return output
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("min_support",type=float,help="Minimum support threshold")
    parser.add_argument("input_file",type=str,help="Input file path")
    parser.add_argument("output_file",type=str,help="Output file path")
    args = parser.parse_args()
    
    current_time = time.time()  
    with open(args.input_file,'r') as f:
        transactions = [list(map(int,line.strip().split(','))) for line in f]
        patterns = fp_growth(transactions,args.min_support)
            
        output = format_output(patterns)
        with open(args.output_file, 'w') as file:
            for line in sorted(output):
                f.write(line + '\n')
        
        execution_time = time.time() - current_time
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"FP-Growth completed with min_support={args.min_support}")
        print(f"Found {len(patterns)} frequent patterns")
        print(f"Results written to {args.output_file}")    