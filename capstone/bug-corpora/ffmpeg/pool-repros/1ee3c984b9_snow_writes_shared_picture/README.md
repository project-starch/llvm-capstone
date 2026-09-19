# snow: modifying a current picture the coded frame still references

The encoder writes into `s->current_picture` while `avctx->coded_frame`
references the same storage. Same class as the filter siblings, on the encoder
side: nothing is freed, and the reader sees its data change identity.

    arm=fixed holder=snow shared_when_written=0 consumer_saw=0xA1 consumer_now=0xA1 freed_to_malloc=0
    arm=buggy holder=snow shared_when_written=1 consumer_saw=0xA1 consumer_now=0xB2 freed_to_malloc=0
