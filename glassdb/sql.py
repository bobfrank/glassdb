import ply.lex # sudo pip3 install ply
import ply.yacc
# TODO fix this grammar, seems wrong based on autocomplete...

keywords = set([
    'NULL','CURRENT_TIME','CURRENT_DATE','CURRENT_TIMESTAMP','EXISTS','NOT',
    'DISTINCT','COLLATE','AS','IS','ISNULL','NOTNULL','IN','LIKE','GLOB','MATCH',
    'REGEX','FROM','WHERE','GROUP','BY','HAVING','NATURAL','LEFT','OUTER','RIGHT',
    'INNER','CROSS','JOIN','ON','USING','SELECT','VALUES','UNION','ALL','EXCEPT',
    'INTERSECT','ASC','DESC','ORDER','LIMIT','OFFSET','OR','AND']
 # 'ABORT','ALTER','ASSERTION','AUTHORIZATION','CATALOG','CHARACTER','COLLATION','CONNECTION','CONSTRAINTS','DOMAIN','FAIL','FUNCTION','GRANT','IGNORE','NAMES','PROCEDURE','RAISE','REVOKE','ROLLBACK','SCHEMA','SESSION','TIME','TRANSACTION','TRANSLATION','TRIGGER','ZONE',
 # 'BEGIN','COMMIT','END',
 # 'CREATE','DELETE','DROP','INSERT','REPLACE','SET','TABLE','UPDATE','VIEW',
)

tokens = ('IDENTIFIER','STRING','REAL','INTEGER','ASTERISK','DOT','COMMA','COLON','AT','DOLLAR','QUESTION','PLUS','MINUS','TILDE','STRING_CONCAT','DIV','MOD','NE','BSR','BSL','BIT_AND','BIT_OR','LE','LT','GE','GT','EEQUALS','EQUALS','OPEN_PAREN','CLOSE_PAREN')+tuple(keywords)

def t_NEWLINE(t):
    r'\n[ ]*'
    t.lexer.lineno += t.value.count('\n')
def t_COMMENT(t):
    r'--.*'
    pass
def t_WHITESPACE(t):
    r'[ \t]+'
    pass
def t_IDENTIFIER(t):
    r'([A-Za-z][A-Za-z0-9_]*|`(?:\\.|[^`\\])*`)'
    if t.value.startswith('`'):
        t.value = t.value[1:-1]
    if t.value.upper() in keywords:
        t.type = t.value.upper()
    return t
t_STRING = r'(\'[^\']*\'|"[^"]*")'
#t_QUOTE = r'"'
t_REAL = r'[0-9]+\.[0-9]+'
t_INTEGER = r'[0-9]+'
t_ASTERISK = r'\*'
t_DOT = r'\.'
t_COMMA = r','
t_COLON = r':'
t_AT = r'@'
t_DOLLAR = r'\$'
t_QUESTION = r'\?'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TILDE = r'~'
t_STRING_CONCAT = r'\|\|'
t_DIV = r'/'
t_MOD = r'%'
t_NE = r'(!=|<>)'
t_BSR = r'>>'
t_BSL = r'<<'
t_BIT_AND = r'&'
t_BIT_OR = r'\|'
t_LE = r'<='
t_LT = r'<'
t_GE = r'>='
t_GT = r'>'
t_EEQUALS = r'=='
t_EQUALS = r'='
t_OPEN_PAREN = r'\('
t_CLOSE_PAREN = r'\)'
#t_END = r';'

def t_error(t):
    print('error! %s'%t)

# parsing

def p_root(p):
    'root : select'
    p[0] = p[1]

def p_select_limit_1(p):
    'select : ordered_select LIMIT expression COMMA expression'
    p[0] = ('limit',p[3],('offset',p[5],p[1]))

def p_select_limit_2(p):
    'select : ordered_select LIMIT expression OFFSET expression'
    p[0] = ('limit',p[3],('offset',p[5],p[1]))

def p_select_limit_3(p):
    'select : ordered_select LIMIT expression'
    p[0] = ('limit',p[3],p[1])

def p_select_next(p):
    'select : ordered_select'
    p[0] = p[1]


def p_ordered_select_order(p): # TODO this wont work as is
    'ordered_select : compound_select ORDER BY ordering_terms'
    p[0] = ('order',p[4],p[1])

def p_ordered_select_next(p):
    'ordered_select : compound_select'
    p[0] = p[1]


def p_compound_select_compound(p):
    'compound_select : compound_select compound_operator select_core'
    p[0] = (p[2],p[1],p[3])

def p_compound_select_next(p):
    'compound_select : select_core'
    p[0] = p[1]


def p_compound_operator_union(p):
    'compound_operator : UNION'
    p[0] = 'union'

def p_compound_operator_unionall(p):
    'compound_operator : UNION ALL'
    p[0] = 'union all'

def p_compound_operator_except(p):
    'compound_operator : EXCEPT'
    p[0] = 'except'

def p_compound_operator_intersect(p):
    'compound_operator : INTERSECT'
    p[0] = 'intersect'


def p_select_core_values(p):
    'select_core : VALUES expression_table'
    p[0] = ('values',p[2])

def p_select_core_next(p):
    'select_core : select_grouped'
    p[0] = p[1]


def p_select_grouped_groupby(p):
    'select_grouped : select_filtered GROUP BY expression_list'
    p[0] = ('group',p[4],p[1])

def p_select_grouped_having(p):
    'select_grouped : select_filtered GROUP BY expression_list HAVING expression'
    p[0] = ('having',('group',p[4],p[1]),p[5])

def p_select_grouped_next(p):
    'select_grouped : select_filtered'
    p[0] = p[1]


def p_select_filtered_where(p):
    'select_filtered : select_from WHERE expression'
    p[0] = ('where',p[3],p[1])

def p_select_filtered_next(p):
    'select_filtered : select_from'
    p[0] = p[1]


def p_select_from_from(p):
    'select_from : select_distinct FROM from'
    p[0] = ('from',p[3],p[1])

def p_select_from_next(p):
    'select_from : select_distinct'
    p[0] = p[1]


def p_select_distinct_dist(p):
    'select_distinct : SELECT DISTINCT result_columns'
    p[0] = ('select-distinct',p[3])

def p_select_distinct_all(p):
    'select_distinct : SELECT ALL result_columns'
    p[0] = ('select-all',p[3])

def p_select_distinct_normal(p):
    'select_distinct : SELECT result_columns'
    p[0] = ('select',p[2])


def p_from_join(p):
    'from : table_or_subquery join_clause'
    p[0] = ('join',p[2],p[1],p[3])

def p_from_next(p):
    'from : table_or_subquery'
    p[0] = p[1]


def p_join_clause_on(p):
    'join_clause : join_operator table_or_subquery ON expression_list'
    p[0] = ('on',p[1],p[3])

def p_join_clause_using(p):
    'join_clause : join_operator USING OPEN_PAREN column_names CLOSE_PAREN'
    p[0] = ('using',p[1],p[4])

def p_join_clause_next(p):
    'join_clause : join_operator table_or_subquery'
    p[0] = p[1]


def p_join_operator_comma(p):
    'join_operator : COMMA'
    p[0] = 'cross'

def p_join_operator_natural(p):
    'join_operator : NATURAL subjoin_operator JOIN'
    p[0] = ('natural',p[2])

def p_join_operator_subjoin(p):
    'join_operator : subjoin_operator JOIN'
    p[0] = p[1]

def p_join_operator_join(p):
    'join_operator : JOIN'
    p[0] = None


def p_subjoin_operator_left(p):
    'subjoin_operator : LEFT'
    p[0] = 'left'

def p_subjoin_operator_left_outer(p):
    'subjoin_operator : LEFT OUTER'
    p[0] = 'left outer'

def p_subjoin_operator_right(p):
    'subjoin_operator : RIGHT'
    p[0] = 'right'

def p_subjoin_operator_right_outer(p):
    'subjoin_operator : RIGHT OUTER'
    p[0] = 'right outer'

def p_subjoin_operator_outer(p):
    'subjoin_operator : OUTER'
    p[0] = 'outer'

def p_subjoin_operator_inner(p):
    'subjoin_operator : INNER'
    p[0] = 'inner'

def p_subjoin_operator_cross(p):
    'subjoin_operator : CROSS'
    p[0] = 'cross'


def p_column_names_more(p):
    'column_names : IDENTIFIER COMMA column_names'
    p[0] = p[3]
    p[0].insert(0, p[1])

def p_column_names_first(p):
    'column_names : IDENTIFIER'
    p[0] = [p[1]]


#def p_table_or_subquery_call(p): # TODO add arguments (expression list or blank), add as ...
#    'table_or_subquery : table_or_function OPEN_PAREN CLOSE_PAREN'
#    p[0] = (p[1],)
#
#def p_table_or_subquery_call_star(p):
#    'table_or_subquery : table_or_function OPEN_PAREN ASTERISK CLOSE_PAREN'
#    p[0] = (p[1], ['*'])

def p_table_or_subquery_next(p):
    'table_or_subquery : table_or_function'
    p[0] = p[1]

def p_table_or_subquery_int(p):
    'table_or_subquery : INTEGER'
    p[0] = p[1]

def p_table_or_subquery_id(p):
    'table_or_subquery : table_or_function IDENTIFIER'
    p[0] = ('as',p[1],p[2])

def p_table_or_subquery_as(p): # TODO add indexed by / not indexed
    'table_or_subquery : table_or_function AS IDENTIFIER'
    p[0] = ('as',p[1],p[3])

def p_table_or_subquery_select(p):
    'table_or_subquery : OPEN_PAREN select CLOSE_PAREN'
    p[0] = p[2]


def p_table_or_function_dot(p):
    'table_or_function : IDENTIFIER DOT IDENTIFIER'
    p[0] = ('.',p[1],p[3])

def p_table_or_function_id(p):
    'table_or_function : IDENTIFIER'
    p[0] = p[1]


def p_expression_table_more(p):
    'expression_table : expression_row COMMA expression_table'
    p[0] = p[3]
    p[0].insert(0, p[1])

def p_expression_table_row(p):
    'expression_table : expression_row'
    p[0] = [p[1]]


def p_expression_row(p):
    'expression_row : OPEN_PAREN expression_list CLOSE_PAREN'
    p[0] = p[2]


def p_result_columns_rest(p):
    'result_columns : result_column COMMA result_columns'
    p[0] = p[3]
    p[0].insert(0,p[1])

def p_result_columns_first(p):
    'result_columns : result_column'
    p[0] = [p[1]]


def p_result_column_asterisk(p):
    'result_column : ASTERISK'
    p[0] = '*'

def p_result_column_id(p):
    'result_column : IDENTIFIER DOT ASTERISK'
    p[0] = ('.',p[1],'*')

def p_result_column_expr(p):
    'result_column : expression'
    p[0] = p[1]


def p_ordering_terms_rest(p):
    'ordering_terms : ordering_term COMMA ordering_terms'
    p[0] = p[3]
    p[0].insert(0, p[1])

def p_ordering_terms_first(p):
    'ordering_terms : ordering_term'
    p[0] = [p[1]]


def p_ordering_term_asc(p):
    'ordering_term : ordering_subterm ASC'
    p[0] = (p[1],'!asc')

def p_ordering_term_desc(p):
    'ordering_term : ordering_subterm DESC'
    p[0] = (p[1],'!desc')

def p_ordering_term_next(p):
    'ordering_term : ordering_subterm'
    p[0] = p[1]


def p_ordering_subterm_collate(p):
    'ordering_subterm : expression COLLATE IDENTIFIER'
    p[0] = ('collate',p[1],p[3])

def p_ordering_subterm_next(p):
    'ordering_subterm : expression'
    p[0] = p[1]


def p_expression_list_rest(p):
    'expression_list : expression COMMA expression_list'
    p[0] = p[3]
    p[0].insert(0, p[1])

def p_exrpression_list_first(p):
    'expression_list : expression'
    p[0] = [p[1]]

def p_expression_isnull(p):
    'expression : expr_binary_or ISNULL'
    p[0] = ('isnull',p[1])

def p_expression_notnull(p):
    'expression : expr_binary_or NOTNULL'
    p[0] = ('notnull',p[1])

def p_expression_is(p):
    'expression : expr_binary_or IS expression'
    p[0] = ('is',p[1],p[3])

def p_expression_base(p):
    'expression : expr_binary_or'
    p[0] = p[1]

def p_expr_binary_or_or(p):
    'expr_binary_or : expr_binary_and OR expr_binary_or'
    p[0] = ('or',p[1],p[3])

def p_expr_binary_or_next(p):
    'expr_binary_or : expr_binary_and'
    p[0] = p[1]


def p_expr_binary_and_and(p):
    'expr_binary_and : expr_binary_streq AND expr_binary_and'
    p[0] = ('and',p[1],p[3])

def p_expr_binary_and_next(p):
    'expr_binary_and : expr_binary_streq'
    p[0] = p[1]


def p_expr_binary_streq_eq(p):
    'expr_binary_streq : expr_binary_ineq EQUALS expr_binary_streq'
    p[0] = ('==',p[1],p[3])

def p_expr_binary_streq_eeq(p):
    'expr_binary_streq : expr_binary_ineq EEQUALS expr_binary_streq'
    p[0] = ('==',p[1],p[3])

def p_expr_binary_streq_ne(p):
    'expr_binary_streq : expr_binary_ineq NE expr_binary_streq'
    p[0] = ('!=',p[1],p[3])

#def p_expr_binary_streq_is(p):
    #'expr_binary_streq : expr_binary_streq IS expr_binary_ineq'
    #p[0] = ('is',p[1],p[3])

#def p_expr_binary_streq_isnot(p):
    #'expr_binary_streq : expr_binary_streq IS NOT expr_binary_ineq'
    #p[0] = ('isnot',p[1],p[3])

def p_expr_binary_streq_in(p):
    'expr_binary_streq : expr_binary_ineq IN expr_binary_streq'
    p[0] = ('in',p[1],p[3])

def p_expr_binary_streq_notin(p):
    'expr_binary_streq : expr_binary_ineq NOT IN expr_binary_streq'
    p[0] = ('notin',p[1],p[3])

def p_expr_binary_streq_like(p):
    'expr_binary_streq : expr_binary_ineq LIKE expr_binary_streq'
    p[0] = ('like',p[1],p[3])

def p_expr_binary_streq_notlike(p):
    'expr_binary_streq : expr_binary_ineq NOT LIKE expr_binary_streq'
    p[0] = ('notlike',p[1],p[3])

def p_expr_binary_streq_glob(p):
    'expr_binary_streq : expr_binary_ineq GLOB expr_binary_streq'
    p[0] = ('glob',p[1],p[3])

def p_expr_binary_streq_match(p):
    'expr_binary_streq : expr_binary_ineq MATCH expr_binary_streq'
    p[0] = ('match',p[1],p[3])

def p_expr_binary_streq_regex(p):
    'expr_binary_streq : expr_binary_ineq REGEX expr_binary_streq'
    p[0] = ('regex',p[1],p[3])

def p_expr_binary_streq_next(p):
    'expr_binary_streq : expr_binary_ineq'
    p[0] = p[1]


def p_expr_binary_ineq_le(p):
    'expr_binary_ineq : expr_binary_bitwise LE expr_binary_ineq'
    p[0] = ('<=',p[1],p[3])

def p_expr_binary_ineq_lt(p):
    'expr_binary_ineq : expr_binary_bitwise LT expr_binary_ineq'
    p[0] = ('<',p[1],p[3])

def p_expr_binary_ineq_gt(p):
    'expr_binary_ineq : expr_binary_bitwise GT expr_binary_ineq'
    p[0] = ('>',p[1],p[3])

def p_expr_binary_ineq_ge(p):
    'expr_binary_ineq : expr_binary_bitwise GE expr_binary_ineq'
    p[0] = ('>=',p[1],p[3])

def p_expr_binary_ineq_next(p):
    'expr_binary_ineq : expr_binary_bitwise'
    p[0] = p[1]


def p_expr_binary_bitwise_bsl(p):
    'expr_binary_bitwise : expr_binary_sum BSL expr_binary_bitwise'
    p[0] = ('<<',p[1],p[3])

def p_expr_binary_bitwise_bsr(p):
    'expr_binary_bitwise : expr_binary_sum BSR expr_binary_bitwise'
    p[0] = ('>>',p[1],p[3])

def p_expr_binary_bitwise_and(p):
    'expr_binary_bitwise : expr_binary_sum BIT_AND expr_binary_bitwise'
    p[0] = ('&',p[1],p[3])

def p_expr_binary_bitwise_or(p):
    'expr_binary_bitwise : expr_binary_sum BIT_OR expr_binary_bitwise'
    p[0] = ('|',p[1],p[3])

def p_expr_binary_bitwise_next(p):
    'expr_binary_bitwise : expr_binary_sum'
    p[0] = p[1]


def p_expr_binary_sum_plus(p):
    'expr_binary_sum : expr_binary_prod PLUS expr_binary_sum'
    p[0] = ('+',p[1],p[3])

def p_expr_binary_sum_minus(p):
    'expr_binary_sum : expr_binary_prod MINUS expr_binary_sum'
    p[0] = ('-',p[1],p[3])

def p_expr_binary_sum_next(p):
    'expr_binary_sum : expr_binary_prod'
    p[0] = p[1]


def p_expr_binary_prod_asterisk(p):
    'expr_binary_prod : expr_binary_stradd ASTERISK expr_binary_prod'
    p[0] = ('*',p[1],p[3])

def p_expr_binary_prod_div(p):
    'expr_binary_prod : expr_binary_stradd DIV expr_binary_prod'
    p[0] = ('/',p[1],p[3])

def p_expr_binary_prod_mod(p):
    'expr_binary_prod : expr_binary_stradd MOD expr_binary_prod'
    p[0] = ('%',p[1],p[3])

def p_expr_binary_prod_next(p):
    'expr_binary_prod : expr_binary_stradd'
    p[0] = p[1]


def p_expr_binary_stradd_concat(p):
    'expr_binary_stradd : expr_unitary STRING_CONCAT expr_binary_stradd'
    p[0] = ('string_concat',p[1],p[3])

def p_expr_binary_stradd_next(p):
    'expr_binary_stradd : expr_unitary'
    p[0] = p[1]

def p_expr_unitary_plus(p):
    'expr_unitary : PLUS expr_core'
    p[0] = p[2]

def p_expr_unitary_minus(p):
    'expr_unitary : MINUS expr_core'
    p[0] = ('-',p[2])

def p_expr_unitary_tilde(p):
    'expr_unitary : TILDE expr_core'
    p[0] = ('~',p[2])

def p_expr_unitary_not(p):
    'expr_unitary : NOT expr_core'
    p[0] = ('!',p[2])

def p_expr_unitary_base(p):
    'expr_unitary : expr_core'
    p[0] = p[1]


def p_expr_core_literal(p):
    'expr_core : literal'
    p[0] = p[1]

def p_expr_core_bind(p):
    'expr_core : bind'
    p[0] = p[1]

def p_expr_core_exists(p):
    'expr_core : EXISTS OPEN_PAREN select CLOSE_PAREN'
    p[0] = ('exists',p[3])

def p_expr_core_parens(p):
    'expr_core : OPEN_PAREN select CLOSE_PAREN'
    p[0] = p[2]

def p_expr_core_stc(p):
    'expr_core : IDENTIFIER DOT IDENTIFIER DOT IDENTIFIER' # schema.table.column
    p[0] = ('.',p[1],p[3],p[5])

def p_expr_core_tc(p):
    'expr_core : IDENTIFIER DOT IDENTIFIER'
    p[0] = ('.',p[1],p[3])

def p_expr_core_id(p):
    'expr_core : IDENTIFIER'
    p[0] = ('.',p[1])

# TODO TOP, DISTINCT, etc
def p_expr_core_func(p):
    'expr_core : IDENTIFIER OPEN_PAREN CLOSE_PAREN'
    p[0] = (p[1],)

def p_expr_core_func_star(p):
    'expr_core : IDENTIFIER OPEN_PAREN ASTERISK CLOSE_PAREN'
    p[0] = (p[1],'*')

def p_expr_core_func_list(p):
    'expr_core : IDENTIFIER OPEN_PAREN expression_list CLOSE_PAREN'
    p[0] = (p[1],p[3])


def p_expr_core_exprlist(p):
    'expr_core : OPEN_PAREN expression_list CLOSE_PAREN'
    p[0] = p[2]


# TODO raise, case, cast

def p_literal_int(p):
    'literal : INTEGER'
    p[0] = int(p[1])

def p_literal_real(p):
    'literal : REAL'
    p[0] = float(p[1])

def p_literal_str(p):
    'literal : STRING'
    p[0] = p[1][1:-1].encode('utf-8')

def p_literal_null(p):
    'literal : NULL'
    p[0] = None

def p_literal_cd(p):
    'literal : CURRENT_DATE'
    p[0] = ('current_date',)

def p_literal_ct(p):
    'literal : CURRENT_TIME'
    p[0] = ('current_time',)

def p_literal_cts(p):
    'literal : CURRENT_TIMESTAMP'
    p[0] = ('current_timestamp',)


# TODO BLOBs

#def p_bind_question(p):
#    'bind : QUESTION'
#    p[0] = (':',) # TODO add the index based on order in query text

def p_bind_qid(p):
    'bind : QUESTION IDENTIFIER'
    p[0] = (':',p[2])

def p_bind_ci(p):
    'bind : COLON IDENTIFIER'
    p[0] = (':',p[2])

def p_bind_at(p):
    'bind : AT IDENTIFIER'
    p[0] = (':',p[2])

def p_bind_dollar(p):
    'bind : DOLLAR IDENTIFIER'
    p[0] = (':',p[2])


def p_error(p):
    raise SyntaxError("Syntax error in input: %s (near '%s')"%(p, p.lexer.lexdata[p.lexpos:p.lexpos+16]))

lexer = ply.lex.lex()
parser = ply.yacc.yacc(tabmodule='sqlparsetab')

def _sql(text):
    return parser.parse(text, lexer)

def sql_completion(text):
    from .sqlparsetab import _lr_action
    import string
    import re
    lexer.input(text)
    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens.append(tok)
    state = 0
    available_tokens = []
    started = ''
    next_ok = False
    for k, tok in enumerate(tokens):
        if k == len(tokens)-1:
            available_tokens = [key for key in _lr_action[state].keys()]
            started = tok.value
        if tok.type in _lr_action[state]:
            state = abs(_lr_action[state][tok.type])
            if k == len(tokens)-1:
                next_ok = True
    available_tokens_next = list(_lr_action[state].keys()) if next_ok else []
    # TODO would be nice to know if the identifier is a column or a table name
    completion = {}
    for av, is_next in [(available_tokens,False), (available_tokens_next,True)]:
        if is_next:
            prev_token = tokens[-1].type
        else:
            if len(tokens) > 1:
                prev_token = tokens[-2].type
            else:
                prev_token = None
        for tok in av:
            if tok != '$end':
                if tok in keywords:
                    if is_next:
                        completion[tok] = (is_next, False, [tok])
                    else:
                        if tok.startswith(started.upper()):
                            completion[tok] = (is_next, False, [tok])
                else:
                    regex = globals()['t_%s'%tok]
                    if not isinstance(regex, str):
                        regex = regex.__doc__
                    tok_re = re.compile(regex)
                    available_chars = []
                    for c in string.printable:
                        if is_next:
                            if tok_re.fullmatch(c):
                                available_chars.append(c)
                        else:
                            if tok_re.fullmatch(started + c):
                                available_chars.append(started + c)
                    if len(available_chars):
                        is_table = tok == 'IDENTIFIER' and prev_token in ['FROM','JOIN']
                        completion[tok] = (is_next, is_table, available_chars)
    return completion


if __name__ == '__main__':
    import time
    s = time.time()
    #pt = _sql('select count(*), test,* from magic left join partial on l=r where a=:a group by xyz order by other')
    sql_completion('select a from')
    #pt = _sql('select count(*), test.* from magic where a=:a group by xyz order by other')
    #print('time',time.time()-s)
    #print(pt)
