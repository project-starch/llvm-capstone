-- Wireshark #16807 — cross-domain use-after-free of a C tvbuff via a Lua TvbRange.
--
-- buffer(0,16) creates a Lua TvbRange userdata that WRAPS the C `tvbuff` of the
-- current dissection. handle_packet() stashes that TvbRange into the global
-- `ProtocolState` table (keyed by packet number). The Lua handle then outlives
-- the C tvbuff: when the packet is re-dissected (GUI packet switch, or two-pass
-- analysis), the previous dissection's tvbuff has been freed by the C engine
-- (epan_dissect_reset -> tvb_free_chain), but ProtocolState still holds the
-- stale TvbRange. subtree:add(foo_field, staleRange) then feeds the dangling
-- tvb straight into proto_tree_add_item_new -> tvb_ensure_bytes_exist -> UAF.
--
-- Two distinct allocations: the Lua TvbRange userdata and the separately
-- g_malloc'd C tvbuff. Unambiguous CDP (native-frees).
--
-- Verbatim from upstream issue #16807 (attachment trigger.lua).

SomeProto = Proto("someproto", "Blah-blah-blah")
AnyTcpPort = 21

SomeProto.fields = {}
SomeProto.fields.Foo = ProtoField.guid("someproto.foo", "Foo field")

ProtocolState = {}

local function handle_packet(packet_id, buffer)
    ProtocolState[packet_id] = {}
    ProtocolState[packet_id][SomeProto.fields.Foo] = buffer(0,16)
end

function SomeProto.dissector(buffer, pinfo, tree)
    if buffer:len() < 16 then return end

    pinfo.cols.protocol = SomeProto.name

    local current_packet_id = pinfo.number
    if ProtocolState[current_packet_id] == nil then
        handle_packet(current_packet_id, buffer)
    end

    local subtree = tree:add(SomeProto, buffer(), "Some Protocol Data")
    local foo_field = SomeProto.fields.Foo
    subtree:add(foo_field, ProtocolState[current_packet_id][foo_field])
end

DissectorTable.get("tcp.port"):add(AnyTcpPort, SomeProto)
