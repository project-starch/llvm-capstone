-- FIXED control for Wireshark #16807.
--
-- Same protocol and same ProtoField.guid, but the field is filled from the
-- CURRENT live `buffer` on every dissection instead of a TvbRange cached in a
-- global table. No Lua handle outlives its C tvbuff, so re-dissection (two-pass
-- or GUI packet-switch) touches only live memory -> no UAF.
--
-- This is the correct pattern: a TvbRange is valid only for the duration of the
-- dissection call that produced it; to persist data across packets, copy the
-- bytes out (:bytes()/:string()), never retain the TvbRange itself.

SomeProto = Proto("someproto", "Blah-blah-blah")
AnyTcpPort = 21

SomeProto.fields = {}
SomeProto.fields.Foo = ProtoField.guid("someproto.foo", "Foo field")

function SomeProto.dissector(buffer, pinfo, tree)
    if buffer:len() < 16 then return end

    pinfo.cols.protocol = SomeProto.name

    local subtree = tree:add(SomeProto, buffer(), "Some Protocol Data")
    subtree:add(SomeProto.fields.Foo, buffer(0,16))   -- always the live buffer
end

DissectorTable.get("tcp.port"):add(AnyTcpPort, SomeProto)
