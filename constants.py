TARGET_COLUMN_ORIGINAL_NAME = 'due_in_date' # Eredeti cél oszlop neve
TARGET_COLUMN_NAME = 'payment_delay_from_due' # Új cél oszlop neve
BASE_COLUMN_NAME = 'clear_date' # Bázis oszlop neve
LABELING_COLUMNS = ['business_code', 'name_customer','invoice_currency','document_type','cust_payment_terms','cust_number'] # Kategóriák kódolásához használt oszlopok
SCALED_MIN = 0.01 # MinMax skálázás minimum értéke
SCALED_MAX = 1 # MinMax skálázás maximum értéke